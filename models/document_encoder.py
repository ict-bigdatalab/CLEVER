import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import BertModel, BertTokenizer
from typing import List, Tuple


class DocumentEncoder(nn.Module):
    """
    Document encoder for CLEVER using BERT.
    
    This class handles document encoding and implements the bootstrapped
    training process with contrastive loss for better representations.
    """
    
    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        dim: int = 768,
        alpha: float = 4.0,
        beta: float = 2.0,
        num_spans_per_granularity: int = 5,
        temperature: float = 0.1,
        phrase_min_length: int = 4,
        phrase_max_length: int = 16,
        sentence_min_length: int = 16,
        sentence_max_length: int = 64,
        paragraph_min_length: int = 64,
        paragraph_max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the document encoder.
        
        Args:
            bert_model_name: The name of the BERT model to use
            dim: Dimension of document representations
            alpha: Alpha parameter for beta distribution in span sampling (skews toward longer spans)
            beta: Beta parameter for beta distribution in span sampling
            num_spans_per_granularity: Number of spans to sample per granularity level
            temperature: Temperature parameter for contrastive loss
            phrase_min_length: Minimum length for phrase-level spans
            phrase_max_length: Maximum length for phrase-level spans
            sentence_min_length: Minimum length for sentence-level spans
            sentence_max_length: Maximum length for sentence-level spans
            paragraph_min_length: Minimum length for paragraph-level spans
            paragraph_max_length: Maximum length for paragraph-level spans
            device: Device to run the model on
        """
        super(DocumentEncoder, self).__init__()
        
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.num_spans_per_granularity = num_spans_per_granularity
        self.temperature = temperature
        self.device = device
        
        # Span length parameters
        self.phrase_min_length = phrase_min_length
        self.phrase_max_length = phrase_max_length
        self.sentence_min_length = sentence_min_length
        self.sentence_max_length = sentence_max_length
        self.paragraph_min_length = paragraph_min_length
        self.paragraph_max_length = paragraph_max_length
        
        # Initialize BERT encoder and tokenizer
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Initialize projector network (feed-forward with tanh activation)
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )
        
        # Move model to device
        self.to(device)
    
    def encode_document(self, document: str) -> torch.Tensor:
        """
        Encode a document using the BERT encoder and projector.
        
        Args:
            document: The document text
            
        Returns:
            The document representation
        """
        # Tokenize and encode the document
        inputs = self.tokenizer(
            document, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get the [CLS] representation
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_representation = outputs.last_hidden_state[:, 0, :]
        
        # Project the representation
        document_representation = self.projector(cls_representation)
        
        return document_representation
    
    def encode_documents_batch(self, documents: List[str]) -> torch.Tensor:
        """
        Encode multiple documents at once.
        
        Args:
            documents: List of document texts
            
        Returns:
            Tensor of document representations
        """
        doc_representations = []
        for doc in documents:
            doc_rep = self.encode_document(doc)
            doc_representations.append(doc_rep)
        
        return torch.stack(doc_representations)
    
    def train_with_contrastive_loss(
        self, 
        documents: List[str], 
        codebook: dict, 
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 16
    ):
        """
        Train the document encoder using contrastive loss and clustering loss.
        
        Args:
            documents: List of documents
            codebook: Current codebook for clustering loss
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size
        """
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=learning_rate
        )
        
        # Training loop
        for epoch in range(num_epochs):
            # Shuffle documents
            indices = list(range(len(documents)))
            random.shuffle(indices)
            
            # Batch training
            for i in range(0, len(documents), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_docs = [documents[idx] for idx in batch_indices]
                
                # Compute contrastive loss
                contrastive_loss = self._compute_contrastive_loss(batch_docs)
                
                # Compute clustering loss
                clustering_loss = self._compute_clustering_loss(batch_docs, codebook)
                
                # Combined loss
                loss = contrastive_loss + clustering_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    def _compute_contrastive_loss(self, documents: List[str]) -> torch.Tensor:
        """
        Compute the contrastive loss for a batch of documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Contrastive loss
        """
        # Sample spans at four levels of granularity
        all_spans = []
        all_span_reps = []
        
        for doc in documents:
            doc_spans = self._sample_spans(doc)
            all_spans.extend(doc_spans)
            
            # Encode spans
            for span in doc_spans:
                span_rep = self._encode_span(span)
                all_span_reps.append(span_rep)
        
        # Encode full documents
        doc_reps = []
        for doc in documents:
            doc_rep = self.encode_document(doc)
            doc_reps.append(doc_rep)
        
        # Combine document and span representations
        all_reps = torch.cat([torch.stack(doc_reps), torch.stack(all_span_reps)], dim=0)
        
        # Compute contrastive loss
        loss = 0.0
        num_docs = len(documents)
        num_spans_per_doc = len(all_spans) // num_docs
        
        for i in range(num_docs):
            # Document representation
            doc_rep = all_reps[i]
            
            # Indices of spans from the same document
            span_indices = list(range(num_docs + i * num_spans_per_doc, 
                                      num_docs + (i + 1) * num_spans_per_doc))
            
            # Compute similarity with all other representations
            sim = torch.matmul(doc_rep, all_reps.T) / self.temperature
            
            # Create mask for positive examples (spans from the same document)
            positive_mask = torch.zeros_like(sim)
            positive_mask[span_indices] = 1.0
            
            # Compute log probability of positive examples
            log_prob = sim - torch.logsumexp(sim, dim=0)
            
            # Compute contrastive loss
            loss += -torch.sum(positive_mask * log_prob) / len(span_indices)
        
        return loss / num_docs
    
    def _sample_spans(self, document: str) -> List[str]:
        """
        Sample spans at three levels of granularity.
        
        Args:
            document: The document text
            
        Returns:
            List of span texts
        """
        # Tokenize document
        tokens = document.split()
        n = len(tokens)
        
        # Define granularity levels as specified in the paper
        levels = [
            (self.phrase_min_length, self.phrase_max_length),      # phrase-level: 4-16
            (self.sentence_min_length, self.sentence_max_length),  # sentence-level: 16-64
            (self.paragraph_min_length, self.paragraph_max_length), # paragraph-level: 64-128
        ]
        
        spans = []
        
        # Sample spans for each level
        for level_min, level_max in levels:
            for _ in range(self.num_spans_per_granularity):
                # Sample span length from beta distribution with alpha=4, beta=2
                # This skews sampling towards longer spans
                p_span = np.random.beta(self.alpha, self.beta)
                span_length = int(p_span * (level_max - level_min) + level_min)
                span_length = min(span_length, n)
                
                # Sample starting position
                if n - span_length <= 0:
                    start = 0
                else:
                    start = random.randint(0, n - span_length)
                
                end = start + span_length
                
                # Extract span
                span = " ".join(tokens[start:end])
                spans.append(span)
        
        return spans
    
    def _encode_span(self, span: str) -> torch.Tensor:
        """
        Encode a span using the BERT encoder.
        
        Args:
            span: The span text
            
        Returns:
            The span representation
        """
        # Tokenize and encode the span
        inputs = self.tokenizer(
            span, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get the word representations
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Average pooling
        attention_mask = inputs['attention_mask']
        span_representation = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)
        
        return span_representation.squeeze(0)
    
    def _compute_clustering_loss(self, documents: List[str], codebook: dict) -> torch.Tensor:
        """
        Compute the clustering loss for a batch of documents.
        
        Args:
            documents: List of documents
            codebook: Current codebook
            
        Returns:
            Clustering loss
        """
        # Encode documents
        doc_reps = []
        for doc in documents:
            doc_rep = self.encode_document(doc)
            doc_reps.append(doc_rep)
        
        doc_reps = torch.stack(doc_reps)
        
        # Split into sub-vectors
        sub_vectors = torch.split(
            doc_reps, 
            self.dim // len(codebook), 
            dim=1
        )
        
        # Compute quantized representations
        quantized_reps = []
        
        for i in range(doc_reps.shape[0]):
            quantized_sub_vectors = []
            
            for m in range(len(codebook)):
                sub_vector = sub_vectors[m][i]
                
                # Find the nearest centroid
                distances = torch.norm(codebook[m] - sub_vector.unsqueeze(0), dim=1)
                k = torch.argmin(distances).item()
                
                # Quantized sub-vector
                quantized_sub_vector = codebook[m][k]
                quantized_sub_vectors.append(quantized_sub_vector)
            
            # Concatenate quantized sub-vectors
            quantized_rep = torch.cat(quantized_sub_vectors)
            quantized_reps.append(quantized_rep)
        
        quantized_reps = torch.stack(quantized_reps)
        
        # Compute MSE loss
        mse_loss = F.mse_loss(doc_reps, quantized_reps)
        
        return mse_loss 