import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import random

class IncrementalProductQuantization:
    """
    Incremental Product Quantization (IPQ) for document encoding
    
    This class implements the IPQ technique described in the CLEVER paper,
    which is designed to efficiently update centroids for new documents.
    """
    
    def __init__(
        self,
        dim: int = 768,
        num_subquantizers: int = 8,
        num_clusters: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the IPQ.
        
        Args:
            dim: Dimension of document representations
            num_subquantizers: Number of sub-quantizers (M)
            num_clusters: Number of clusters per sub-quantizer (K)
            device: Device to run on
        """
        self.dim = dim
        self.num_subquantizers = num_subquantizers
        self.num_clusters = num_clusters
        self.device = device
        
        # Initialize codebook and document clusters
        self.codebook = None
        self.document_clusters = None
        
        # Session counter
        self.session = 0
        
        # New centroids counter
        self.new_centroids_count = None
    
    def initialize_codebook(self, document_representations: torch.Tensor) -> Tuple[Dict, Dict, List[List[int]]]:
        """
        Initialize the codebook and document clusters using K-means clustering.
        
        Args:
            document_representations: Tensor of document representations
            
        Returns:
            Tuple of (codebook, document_clusters)
        """
        # Ensure document_representations is 2D
        if document_representations.dim() == 3:
            batch_size, seq_len, emb_dim = document_representations.size()
            document_representations = document_representations.reshape(batch_size * seq_len, emb_dim)
        
        # Determine the number of clusters based on data size
        n_samples = document_representations.shape[0]
        actual_clusters = min(self.num_clusters, n_samples)
        
        if actual_clusters < self.num_clusters:
            print(f"Warning: Reducing clusters from {self.num_clusters} to {actual_clusters} due to small sample size.")
        
        # Split representation into sub-vectors
        sub_vectors = torch.split(
            document_representations, 
            self.dim // self.num_subquantizers, 
            dim=1
        )
        
        # Initialize codebook and document clusters
        codebook = {}
        document_clusters = {}
        
        # For each sub-quantizer
        for m in range(self.num_subquantizers):
            # Apply K-means clustering
            sub_vectors_np = sub_vectors[m].detach().cpu().numpy()
            kmeans = KMeans(n_clusters=actual_clusters, random_state=0).fit(sub_vectors_np)
            
            # Store centroids
            centroids = torch.tensor(kmeans.cluster_centers_, device=self.device)
            codebook[m] = centroids
            
            # Store document clusters
            document_clusters[m] = {}
            for k in range(actual_clusters):
                document_clusters[m][k] = []
            
            # Assign documents to clusters
            labels = kmeans.labels_
            for i, label in enumerate(labels):
                document_clusters[m][int(label)].append(i)
        
        # Store the codebook and document clusters
        self.codebook = codebook
        self.document_clusters = document_clusters
        self.session = 0
        
        # Initialize new centroids counter
        self.new_centroids_count = {m: 0 for m in range(self.num_subquantizers)}
        
        # Generate PQ codes for all base documents
        pq_codes = self.generate_pq_codes(document_representations)
        
        return self.codebook, self.document_clusters, pq_codes
    
    def generate_pq_codes(self, document_representations: torch.Tensor) -> List[List[int]]:
        """
        Generate PQ codes for document representations.
        
        Args:
            document_representations: Tensor of document representations (batch_size, dim) or (batch_size, seq_len, dim)
            
        Returns:
            List of PQ codes, where each PQ code is a list of cluster indices
        """
        if self.codebook is None:
            raise ValueError("Codebook is not initialized. Call initialize_codebook first.")
        
        # Ensure document_representations is 2D
        if document_representations.dim() == 3:
            batch_size, seq_len, emb_dim = document_representations.size()
            document_representations = document_representations.reshape(batch_size * seq_len, emb_dim)
        
        pq_codes = []
        
        # Split representations into sub-vectors
        sub_vectors = torch.split(
            document_representations, 
            self.dim // self.num_subquantizers, 
            dim=1
        )
        
        # For each document
        for i in range(document_representations.shape[0]):
            doc_pq_code = []
            
            # For each sub-quantizer
            for m in range(self.num_subquantizers):
                # Get sub-vector for the document
                sub_vector = sub_vectors[m][i]
                
                # Find the nearest centroid
                distances = torch.norm(self.codebook[m] - sub_vector.unsqueeze(0), dim=1)
                k = torch.argmin(distances).item()
                
                doc_pq_code.append(k)
            
            pq_codes.append(doc_pq_code)
        
        return pq_codes
    
    def update_codebook(
        self, 
        document_representations: torch.Tensor,
        document_indices: List[int] = None
    ) -> Tuple[Dict, Dict, List[List[int]]]:
        """
        Adaptively update quantization centroids for new documents.
        
        Args:
            document_representations: Tensor of document representations (batch_size, dim) or (batch_size, seq_len, dim)
            document_indices: Optional list of document indices (for tracking)
            
        Returns:
            Tuple of (updated_codebook, updated_document_clusters, pq_codes)
        """
        # Ensure document_representations is 2D
        if document_representations.dim() == 3:
            batch_size, seq_len, emb_dim = document_representations.size()
            document_representations = document_representations.reshape(batch_size * seq_len, emb_dim)
            
        if self.codebook is None:
            return self.initialize_codebook(document_representations)
        
        self.session += 1
        
        # Split into sub-vectors
        sub_vectors = torch.split(
            document_representations, 
            self.dim // self.num_subquantizers, 
            dim=1
        )
        
        # Create copies of codebook and document clusters
        updated_codebook = {m: self.codebook[m].clone() for m in range(self.num_subquantizers)}
        
        # Get actual clusters count
        actual_clusters = {}
        for m in range(self.num_subquantizers):
            actual_clusters[m] = self.codebook[m].shape[0]
        
        # Initialize new centroids counter if None
        if self.new_centroids_count is None:
            self.new_centroids_count = {m: 0 for m in range(self.num_subquantizers)}
        
        # Create updated document clusters dictionary safely
        updated_document_clusters = {}
        for m in range(self.num_subquantizers):
            updated_document_clusters[m] = {}
            num_clusters = actual_clusters[m] + self.new_centroids_count.get(m, 0)
            for k in range(num_clusters):
                # Only copy if the key exists in document_clusters
                if k in self.document_clusters.get(m, {}):
                    updated_document_clusters[m][k] = self.document_clusters[m][k].copy()
                else:
                    updated_document_clusters[m][k] = []
        
        # PQ codes for new documents
        pq_codes = []
        
        # For each new document
        for doc_idx, doc_rep in enumerate(document_representations):
            doc_pq_code = []
            
            # Get actual document index if provided
            actual_idx = document_indices[doc_idx] if document_indices is not None else doc_idx
            
            # For each sub-quantizer
            for m in range(self.num_subquantizers):
                sub_vector = sub_vectors[m][doc_idx]
                
                # Find the nearest centroid
                distances = torch.norm(self.codebook[m] - sub_vector.unsqueeze(0), dim=1)
                k = torch.argmin(distances).item()
                
                # Compute distance to the nearest centroid
                dist = distances[k].item()
                
                # Compute adaptive thresholds
                if len(self.document_clusters[m][k]) > 0:
                    # Get all document sub-vectors in this cluster
                    cluster_doc_indices = self.document_clusters[m][k]
                    
                    # Compute distances from centroid to all documents in cluster
                    distances_from_centroid = []
                    
                    for idx in cluster_doc_indices:
                        # For simplicity, assume we have access to the original representations
                        # In a real implementation, we would need to store or recompute these
                        doc_sub_vector = sub_vectors[m][idx % document_representations.shape[0]]
                        dist_from_centroid = torch.norm(self.codebook[m][k] - doc_sub_vector).item()
                        distances_from_centroid.append(dist_from_centroid)
                    
                    # Compute average distance (ad)
                    ad = sum(distances_from_centroid) / len(distances_from_centroid)
                    
                    # Compute maximum distance (md)
                    md = max(distances_from_centroid)
                    
                    # Add random distance
                    rand_dist = random.uniform(0, ad)
                    md += rand_dist
                else:
                    # If no documents in the cluster, use default values
                    ad = 0.1
                    md = 0.2
                
                # Update based on thresholds
                if dist < ad:
                    # Unchanged old centroid
                    centroid_idx = k
                    # Still add document to the cluster for future threshold computations
                    updated_document_clusters[m][k].append(actual_idx)
                elif ad <= dist <= md:
                    # Changed old centroid
                    # Update document clusters
                    updated_document_clusters[m][k].append(actual_idx)
                    
                    # Update centroid representation
                    cluster_size = len(updated_document_clusters[m][k])
                    updated_codebook[m][k] = self.codebook[m][k] + (1 / cluster_size) * (sub_vector - self.codebook[m][k])
                    
                    centroid_idx = k
                else:
                    # Add new centroid
                    new_k = actual_clusters[m] + self.new_centroids_count[m]
                    self.new_centroids_count[m] += 1
                    
                    # Add new centroid to codebook
                    new_centroids = torch.cat([updated_codebook[m], sub_vector.unsqueeze(0)], dim=0)
                    updated_codebook[m] = new_centroids
                    
                    # Add new cluster to document clusters
                    updated_document_clusters[m][new_k] = [actual_idx]
                    
                    centroid_idx = new_k
                
                doc_pq_code.append(centroid_idx)
            
            pq_codes.append(doc_pq_code)
        
        # Update codebook and document clusters
        self.codebook = updated_codebook
        self.document_clusters = updated_document_clusters
        
        return self.codebook, self.document_clusters, pq_codes
    
    def get_centroid_for_pq_code(self, pq_code: List[int]) -> torch.Tensor:
        """
        Get the centroids corresponding to a PQ code.
        
        Args:
            pq_code: PQ code (list of cluster indices)
            
        Returns:
            Tensor representation by concatenating the centroids
        """
        if self.codebook is None:
            raise ValueError("Codebook is not initialized. Call initialize_codebook first.")
        
        centroid_vectors = []
        
        for m, k in enumerate(pq_code):
            if k >= len(self.codebook[m]):
                raise ValueError(f"Invalid cluster index {k} for sub-quantizer {m}")
            
            centroid_vectors.append(self.codebook[m][k])
        
        return torch.cat(centroid_vectors)
    
    def compute_quantization_error(self, document_representations: torch.Tensor) -> float:
        """
        Compute the quantization error for a set of document representations.
        
        Args:
            document_representations: Tensor of document representations
            
        Returns:
            Average quantization error
        """
        if self.codebook is None:
            raise ValueError("Codebook is not initialized. Call initialize_codebook first.")
        
        # Generate PQ codes
        pq_codes = self.generate_pq_codes(document_representations)
        
        # Compute quantization error
        total_error = 0.0
        
        for i, pq_code in enumerate(pq_codes):
            # Get original document representation
            original_rep = document_representations[i]
            
            # Get quantized representation
            quantized_rep = self.get_centroid_for_pq_code(pq_code)
            
            # Compute error
            error = torch.norm(original_rep - quantized_rep).item()
            total_error += error
        
        # Compute average error
        avg_error = total_error / len(pq_codes)
        
        return avg_error
    
    def quantize(self, document_representations: torch.Tensor) -> Tuple[Dict, Dict, List[List[int]]]:
        """
        Quantize document representations using the IPQ method.
        
        Args:
            document_representations: Tensor of document representations (batch_size, dim) or (batch_size, seq_len, dim)
            
        Returns:
            Tuple of (codebook, document_clusters, pq_codes)
        """
        # Ensure document_representations is 2D
        if document_representations.dim() == 3:
            batch_size, seq_len, emb_dim = document_representations.size()
            document_representations = document_representations.reshape(batch_size * seq_len, emb_dim)
        
        # If codebook not initialized, initialize it
        if self.codebook is None:
            return self.initialize_codebook(document_representations)
        
        # Otherwise, generate PQ codes using existing codebook
        pq_codes = self.generate_pq_codes(document_representations)
        return self.codebook, self.document_clusters, pq_codes
    
    def get_all_centroids(self):
        """
        Get all centroids for saving or serialization.
        
        Returns:
            A dictionary representation of centroids that can be JSON serialized
        """
        if self.codebook is None:
            return {}
            
        # Convert tensors to lists for JSON serialization
        serializable_codebook = {}
        for m in range(self.num_subquantizers):
            serializable_codebook[str(m)] = self.codebook[m].detach().cpu().numpy().tolist()
            
        return serializable_codebook 