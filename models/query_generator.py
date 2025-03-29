import torch
import random
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Tuple, Dict

class QueryGenerator:
    """
    Query generator for CLEVER using T5 model.
    
    This class generates pseudo-queries for documents to maintain
    the retrieval ability during continual learning.
    """
    
    def __init__(
        self,
        model_name: str = "t5-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512
    ):
        """
        Initialize the query generator.
        
        Args:
            model_name: The name of the T5 model to use
            device: Device to run the model on
            max_length: Maximum length of input documents
        """
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = device
        self.max_length = max_length
        
        # Move model to device
        self.model.to(device)
    
    def finetune(
        self,
        query_document_pairs: List[Tuple[str, str]],
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 1e-4
    ):
        """
        Fine-tune the T5 model on query-document pairs.
        
        Args:
            query_document_pairs: List of (query, document) pairs
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Set model to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            # Shuffle data
            random.shuffle(query_document_pairs)
            
            # Batch training
            for i in range(0, len(query_document_pairs), batch_size):
                batch_pairs = query_document_pairs[i:i+batch_size]
                
                # Prepare inputs and targets
                input_texts = ["generate query: " + doc for _, doc in batch_pairs]
                target_texts = [query for query, _ in batch_pairs]
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    input_texts,
                    padding="longest",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                # Tokenize targets
                with self.tokenizer.as_target_tokenizer():
                    targets = self.tokenizer(
                        target_texts,
                        padding="longest",
                        truncation=True,
                        max_length=64,  # Shorter max length for queries
                        return_tensors="pt"
                    ).to(self.device)
                
                # Replace padding token id with -100 so it's ignored in loss computation
                target_ids = targets.input_ids.clone()
                target_ids[target_ids == self.tokenizer.pad_token_id] = -100
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=target_ids
                )
                
                loss = outputs.loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Set model back to evaluation mode
        self.model.eval()
    
    def generate_queries(
        self,
        documents: List[str],
        num_queries_per_doc: int = 3,
        max_query_length: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> List[Tuple[str, int]]:
        """
        Generate pseudo-queries for documents.
        
        Args:
            documents: List of documents
            num_queries_per_doc: Number of pseudo-queries to generate per document
            max_query_length: Maximum length of generated queries
            temperature: Temperature for generation
            top_p: Top-p sampling parameter
            
        Returns:
            List of (query, document_index) pairs
        """
        # Set model to evaluation mode
        self.model.eval()
        
        pseudo_query_doc_pairs = []
        
        for doc_idx, doc in enumerate(documents):
            # Truncate document if too long
            doc_tokens = doc.split()
            if len(doc_tokens) > self.max_length:
                doc = " ".join(doc_tokens[:self.max_length])
            
            # Prepare input
            input_text = "generate query: " + doc
            
            # Tokenize input
            input_ids = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True
            ).input_ids.to(self.device)
            
            # Generate queries
            with torch.no_grad():
                for _ in range(num_queries_per_doc):
                    outputs = self.model.generate(
                        input_ids,
                        max_length=max_query_length,
                        num_return_sequences=1,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature
                    )
                    
                    query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    pseudo_query_doc_pairs.append((query, doc_idx))
        
        return pseudo_query_doc_pairs
    
    def save_model(self, path: str):
        """
        Save the model and tokenizer.
        
        Args:
            path: Path to save the model to
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load_model(self, path: str):
        """
        Load the model and tokenizer.
        
        Args:
            path: Path to load the model from
        """
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model.to(self.device) 