"""
Character Vocabulary and CTC Decoding Utilities
================================================

This module handles:
1. Character vocabulary management
2. Text encoding/decoding
3. CTC greedy decoding
4. CTC beam search decoding (placeholder for future)

The vocabulary includes:
- Lowercase letters (a-z)
- Uppercase letters (A-Z)
- Digits (0-9)
- Common punctuation and spaces
- CTC blank token (index 0)
"""

import torch
import string
from typing import List, Tuple


class Vocabulary:
    """
    Manages character vocabulary for OCR.
    
    The vocabulary includes a special CTC blank token at index 0,
    followed by all other characters.
    
    Attributes:
        chars: String containing all characters (except blank)
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
        blank_idx: Index of the CTC blank token (always 0)
        num_classes: Total number of classes (including blank)
    """
    
    def __init__(self, custom_chars: str = None):
        """
        Initialize vocabulary.
        
        Args:
            custom_chars: Optional custom character set. If None, uses default set.
        """
        if custom_chars is None:
            # Default character set
            self.chars = (
                string.ascii_lowercase +        # a-z
                string.ascii_uppercase +        # A-Z
                string.digits +                 # 0-9
                string.punctuation +            # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
                ' ' +                           # space
                '“' + '”' +
                '’' +
                'é'
            )
        else:
            self.chars = custom_chars
        
        # CTC blank token is always at index 0
        self.blank_idx = 0
        
        # Create mappings
        # Index 0 is reserved for CTC blank
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = '<blank>'  # For debugging/visualization
        
        # Total number of classes (including blank)
        self.num_classes = len(self.chars) + 1
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text string to list of indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of character indices
            
        Raises:
            ValueError: If text contains unknown characters
        """
        encoded = []
        for char in text:
            if char not in self.char_to_idx:
                raise ValueError(
                    f"Unknown character '{char}' (ord={ord(char)}). "
                    f"Add it to the vocabulary or remove it from the text."
                )
            encoded.append(self.char_to_idx[char])
        return encoded
    
    def decode(self, indices: List[int], remove_duplicates: bool = False) -> str:
        """
        Decode list of indices to text string.
        
        Args:
            indices: List of character indices
            remove_duplicates: If True, removes consecutive duplicate characters
                             (useful for CTC output visualization before collapsing)
            
        Returns:
            Decoded text string
        """
        if remove_duplicates:
            indices = self._remove_consecutive_duplicates(indices)
        
        decoded_chars = []
        for idx in indices:
            if idx == self.blank_idx:
                continue  # Skip blank tokens
            if idx in self.idx_to_char:
                decoded_chars.append(self.idx_to_char[idx])
            else:
                decoded_chars.append('<?>')  # Unknown index
        
        return ''.join(decoded_chars)
    
    def _remove_consecutive_duplicates(self, indices: List[int]) -> List[int]:
        """Remove consecutive duplicate indices."""
        if not indices:
            return []
        
        result = [indices[0]]
        for idx in indices[1:]:
            if idx != result[-1]:
                result.append(idx)
        return result
    
    def __len__(self) -> int:
        """Return total number of classes (including blank)."""
        return self.num_classes
    
    def __repr__(self) -> str:
        """String representation of vocabulary."""
        return (
            f"Vocabulary(num_classes={self.num_classes}, "
            f"num_chars={len(self.chars)}, "
            f"blank_idx={self.blank_idx})"
        )


class CTCDecoder:
    """
    CTC Decoding utilities for converting model outputs to text.
    
    Implements:
    1. Greedy decoding (fast, simple)
    2. Beam search decoding (better accuracy, slower) - TODO
    """
    
    def __init__(self, vocabulary: Vocabulary):
        """
        Initialize CTC decoder.
        
        Args:
            vocabulary: Vocabulary instance for character mapping
        """
        self.vocab = vocabulary
    
    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        input_lengths: torch.Tensor = None
    ) -> List[str]:
        """
        Greedy CTC decoding.
        
        For each timestep, selects the most probable character.
        Then applies CTC collapsing rules:
        1. Remove consecutive duplicate characters
        2. Remove blank tokens
        
        Args:
            log_probs: Log probabilities from model (T, B, C) or (B, T, C)
            input_lengths: Length of each sequence in batch (B,)
                         If None, assumes all sequences are full length
        
        Returns:
            List of decoded text strings (one per batch element)
        """
        # Handle both (T, B, C) and (B, T, C) formats
        if log_probs.dim() == 3:
            if log_probs.size(0) < log_probs.size(1):
                # Likely (B, T, C), transpose to (T, B, C)
                log_probs = log_probs.permute(1, 0, 2)
        
        # Get most probable indices at each timestep
        # Shape: (T, B)
        _, max_indices = torch.max(log_probs, dim=2)
        
        # Convert to numpy for easier processing
        max_indices = max_indices.cpu().numpy()
        
        # Decode each sequence in batch
        batch_size = max_indices.shape[1]
        decoded_texts = []
        
        for b in range(batch_size):
            # Get sequence for this batch element
            sequence = max_indices[:, b].tolist()
            
            # Trim to actual length if provided
            if input_lengths is not None:
                actual_length = input_lengths[b].item()
                sequence = sequence[:actual_length]
            
            # Apply CTC collapsing
            collapsed = self._ctc_collapse(sequence)
            
            # Decode to text
            text = self.vocab.decode(collapsed)
            decoded_texts.append(text)
        
        return decoded_texts
    
    def _ctc_collapse(self, sequence: List[int]) -> List[int]:
        """
        Apply CTC collapsing rules:
        1. Remove consecutive duplicates
        2. Remove blanks
        
        Example:
            [1, 1, 0, 2, 2, 0, 3] → [1, 2, 3]
        
        Args:
            sequence: List of indices
            
        Returns:
            Collapsed sequence without blanks and duplicates
        """
        if not sequence:
            return []
        
        collapsed = []
        previous = None
        
        for idx in sequence:
            # Skip blanks
            if idx == self.vocab.blank_idx:
                previous = None  # Reset previous to allow same char after blank
                continue
            
            # Skip consecutive duplicates
            if idx != previous:
                collapsed.append(idx)
                previous = idx
        
        return collapsed
    
    def beam_search_decode(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 10,
        input_lengths: torch.Tensor = None
    ) -> List[str]:
        """
        Beam search CTC decoding.
        
        TODO: Implement proper beam search with:
        - Prefix beam search algorithm
        - Language model integration (optional)
        - Length normalization
        
        For now, falls back to greedy decoding.
        
        Args:
            log_probs: Log probabilities from model (T, B, C)
            beam_width: Number of beams to maintain
            input_lengths: Length of each sequence in batch (B,)
        
        Returns:
            List of decoded text strings
        """
        print("Warning: Beam search not yet implemented. Using greedy decoding.")
        return self.greedy_decode(log_probs, input_lengths)


def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Character Error Rate (CER).
    
    CER = (Substitutions + Insertions + Deletions) / Total Characters in Target
    
    Uses Levenshtein distance (edit distance) between strings.
    
    Args:
        predictions: List of predicted text strings
        targets: List of ground truth text strings
    
    Returns:
        CER as a percentage (0-100)
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        distance = levenshtein_distance(pred, target)
        total_distance += distance
        total_length += len(target)
    
    if total_length == 0:
        return 0.0
    
    cer = (total_distance / total_length) * 100
    return cer


def calculate_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).
    
    WER = (Substitutions + Insertions + Deletions) / Total Words in Target
    
    Args:
        predictions: List of predicted text strings
        targets: List of ground truth text strings
    
    Returns:
        WER as a percentage (0-100)
    """
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        distance = levenshtein_distance(pred_words, target_words)
        total_distance += distance
        total_words += len(target_words)
    
    if total_words == 0:
        return 0.0
    
    wer = (total_distance / total_words) * 100
    return wer


def levenshtein_distance(s1, s2) -> int:
    """
    Calculate Levenshtein distance (edit distance) between two sequences.
    
    Works with both strings and lists of strings (for WER).
    
    Args:
        s1: First sequence
        s2: Second sequence
    
    Returns:
        Edit distance (number of operations needed to transform s1 into s2)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def test_vocabulary():
    """Test vocabulary encoding and decoding."""
    print("=" * 70)
    print("Vocabulary Testing")
    print("=" * 70)
    
    vocab = Vocabulary()
    print(f"\n{vocab}")
    print(f"Sample characters: {vocab.chars[:20]}...")
    print(f"Blank index: {vocab.blank_idx}")
    
    # Test encoding
    test_text = "Hello, World! 123"
    print(f"\nOriginal text: '{test_text}'")
    
    encoded = vocab.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = vocab.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    assert decoded == test_text, "Encoding/decoding mismatch!"
    print("✓ Encoding/decoding test passed!")
    
    # Test CTC collapsing simulation
    print("\n" + "-" * 70)
    print("CTC Collapsing Simulation")
    print("-" * 70)
    
    # Simulate CTC output with duplicates and blanks
    simulated_ctc = [0, 8, 8, 0, 5, 5, 12, 0, 12, 15, 0]  # "H e l l o" with blanks
    print(f"Simulated CTC output: {simulated_ctc}")
    
    decoded_ctc = vocab.decode(simulated_ctc, remove_duplicates=True)
    print(f"Decoded (with collapsing): '{decoded_ctc}'")
    
    print("\n" + "=" * 70)


def test_ctc_decoder():
    """Test CTC decoder."""
    print("\n" + "=" * 70)
    print("CTC Decoder Testing")
    print("=" * 70)
    
    vocab = Vocabulary()
    decoder = CTCDecoder(vocab)
    
    # Create dummy log probabilities
    # Shape: (T=20, B=2, C=num_classes)
    T, B, C = 20, 2, vocab.num_classes
    log_probs = torch.randn(T, B, C)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=2)
    
    print(f"\nLog probs shape: {log_probs.shape}")
    print(f"  - Sequence length: {T}")
    print(f"  - Batch size: {B}")
    print(f"  - Num classes: {C}")
    
    # Decode
    decoded_texts = decoder.greedy_decode(log_probs)
    
    print(f"\nDecoded texts:")
    for i, text in enumerate(decoded_texts):
        print(f"  Batch {i}: '{text}'")
    
    print("\n✓ CTC decoder test passed!")
    print("=" * 70)


def test_error_metrics():
    """Test CER and WER calculations."""
    print("\n" + "=" * 70)
    print("Error Metrics Testing")
    print("=" * 70)
    
    # Test cases
    predictions = [
        "hello world",
        "the quick brown fox",
        "testing 123"
    ]
    
    targets = [
        "hello world",        # Perfect match
        "the quik brown fox", # 1 char error
        "testing"             # 4 char insertions
    ]
    
    print("\nTest cases:")
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        print(f"\n{i+1}. Target:     '{target}'")
        print(f"   Prediction: '{pred}'")
        edit_dist = levenshtein_distance(pred, target)
        print(f"   Edit distance: {edit_dist}")
    
    # Calculate metrics
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    
    print(f"\n{'='*70}")
    print(f"Overall CER: {cer:.2f}%")
    print(f"Overall WER: {wer:.2f}%")
    print(f"{'='*70}")
    
    print("\n✓ Error metrics test passed!")


if __name__ == "__main__":
    # Run all tests
    test_vocabulary()
    test_ctc_decoder()
    test_error_metrics()