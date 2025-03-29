from typing import List, Dict


class Trie(object):
    """
    Trie for constrained decoding with beam search.
    
    This class implements a prefix tree to constrain the generation of tokens
    during beam search to valid PQ codes.
    """
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        """
        Append another trie to this one.
        
        Args:
            trie: The trie to append
            bos_token_id: The beginning of sequence token ID
        """
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        """
        Add a sequence to the trie.
        
        Args:
            sequence: The token sequence to add
        """
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        """
        Get valid next tokens given a prefix sequence.
        
        Args:
            prefix_sequence: The current sequence of tokens
            
        Returns:
            List of valid next token IDs
        """
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        """
        Load a trie from a dictionary.
        
        Args:
            trie_dict: The dictionary to load from
            
        Returns:
            A Trie object
        """
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        """
        Add a sequence to a trie dictionary.
        
        Args:
            sequence: The token sequence to add
            trie_dict: The trie dictionary to add to
        """
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        """
        Get valid next tokens given a prefix sequence.
        
        Args:
            prefix_sequence: The current sequence of tokens
            trie_dict: The trie dictionary to get from
            append_trie: Optional trie to append
            bos_token_id: The beginning of sequence token ID
            
        Returns:
            List of valid next token IDs
        """
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        """
        Iterate through the trie.
        
        Returns:
            Iterator over sequences in the trie
        """
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        """
        Get the number of sequences in the trie.
        
        Returns:
            The number of sequences in the trie
        """
        return self.len

    def __getitem__(self, value):
        """
        Get valid next tokens for a sequence.
        
        Args:
            value: The sequence to get next tokens for
            
        Returns:
            List of valid next token IDs
        """
        return self.get(value)
