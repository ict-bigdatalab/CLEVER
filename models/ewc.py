import torch
import torch.nn as nn
from typing import List, Dict, Optional

class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.
    
    This class implements EWC as described in the CLEVER paper to prevent
    forgetting of the retrieval ability during continual learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 10.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the EWC.
        
        Args:
            model: The model to apply EWC to
            ewc_lambda: Lambda parameter for the EWC loss
            device: Device to run on
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        
        # Store Fisher information matrix
        self.fisher_information = None
        
        # Store old parameter values
        self.old_params = None
    
    def compute_fisher_information(
        self, 
        loss_fn: callable, 
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 100
    ):
        """
        Compute Fisher information matrix for EWC.
        
        Args:
            loss_fn: Loss function to compute gradients
            data_loader: Data loader for computing Fisher information
            num_samples: Number of samples to use for Fisher computation
        """
        # Get model parameters
        params = list(self.model.parameters())
        
        # Store current parameter values
        self.old_params = [p.clone().detach() for p in params]
        
        # Initialize Fisher information
        self.fisher_information = [torch.zeros_like(p) for p in params]
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Compute Fisher information
        sample_count = 0
        for batch in data_loader:
            if sample_count >= num_samples:
                break
            
            # Forward pass
            loss = loss_fn(self.model, batch)
            
            # Backward pass
            loss.backward()
            
            # Accumulate squared gradients
            for i, p in enumerate(params):
                if p.grad is not None:
                    self.fisher_information[i] += p.grad.data ** 2
                    
            sample_count += 1
        
        # Normalize
        for i in range(len(self.fisher_information)):
            self.fisher_information[i] /= sample_count
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC loss to regularize model parameters.
        
        Returns:
            EWC loss
        """
        if self.fisher_information is None or self.old_params is None:
            return torch.tensor(0.0, device=self.device)
        
        loss = 0.0
        params = list(self.model.parameters())
        
        for i, (p, old_p, fisher) in enumerate(zip(params, self.old_params, self.fisher_information)):
            loss += torch.sum(fisher * (p - old_p) ** 2)
        
        return self.ewc_lambda * loss
    
    def save_state(self, path: str):
        """
        Save the EWC state (Fisher information and old parameters).
        
        Args:
            path: Path to save the state to
        """
        state = {
            'fisher_information': self.fisher_information,
            'old_params': self.old_params,
            'ewc_lambda': self.ewc_lambda
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """
        Load the EWC state (Fisher information and old parameters).
        
        Args:
            path: Path to load the state from
        """
        state = torch.load(path)
        self.fisher_information = state['fisher_information']
        self.old_params = state['old_params']
        self.ewc_lambda = state['ewc_lambda'] 