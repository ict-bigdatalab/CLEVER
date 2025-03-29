import torch
from transformers import LogitsProcessor


class TrieLogitsProcessor(LogitsProcessor):
    """
    Logits processor that constrains generation based on a trie.
    
    This processor filters the logits to only allow tokens that are valid
    next tokens according to the trie.
    """
    def __init__(self, trie, eos_token_id, initial_input_ids=None):
        """
        Initialize the TrieLogitsProcessor.
        
        Args:
            trie: The trie to use for constraining generation
            eos_token_id: The end of sequence token ID
            initial_input_ids: Optional initial input IDs
        """
        self.trie = trie
        self.eos_token_id = eos_token_id
        self.initial_input_ids = initial_input_ids

    def __call__(self, input_ids, scores):
        """
        Process logits to constrain generation.
        
        Args:
            input_ids: The current input IDs
            scores: The logits to process
            
        Returns:
            The processed logits
        """
        batch_size = scores.size(0)
        
        # Process each sequence in the batch
        for i in range(batch_size):
            # Get current sequence
            current_seq = input_ids[i].tolist()
            
            # Create mask for valid tokens
            mask = torch.ones_like(scores[i], dtype=torch.bool)
            
            # Get valid next tokens from trie
            valid_tokens = self.trie.get(current_seq)
            
            # Special handling for EOS token
            if valid_tokens == [self.eos_token_id]:
                mask[self.eos_token_id] = False
            else:
                # Set mask to False for valid tokens
                for token_id in valid_tokens:
                    mask[token_id] = False
            
            # Mask out invalid tokens
            scores[i, mask] = float('-inf')

        return scores
