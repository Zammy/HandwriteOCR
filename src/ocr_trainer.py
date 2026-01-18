"""
OCR Training Utilities with CTC Loss
=====================================

This module provides training and validation functions specifically
designed for OCR models using CTC loss.

Key differences from standard classification:
1. Variable-length sequences (input and target)
2. CTC loss requires special input format
3. Evaluation uses CER/WER instead of accuracy
4. Decoding required to get text predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, List
import time

from .vocab_ctc import CTCDecoder, Vocabulary, calculate_cer, calculate_wer


class OCRTrainer:
    """
    Handles training and validation for OCR models with CTC loss.
    
    Features:
    - CTC loss computation
    - Automatic length calculation
    - CER/WER metric tracking
    - Progress reporting
    """
    
    def __init__(
        self,
        model: nn.Module,
        vocab: Vocabulary,
        device: torch.device,
    ):
        """
        Initialize OCR trainer.
        
        Args:
            model: CRNN model instance
            vocab: Vocabulary instance
            device: Device to train on (cuda/cpu)
            ctc_blank_idx: Index of CTC blank token
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.ctc_blank_idx = vocab.blank_idx
        
        # CTC Loss
        self.ctc_loss = nn.CTCLoss(
            blank=self.ctc_blank_idx,
            reduction='mean',
            zero_infinity=True  # Prevents inf loss with invalid targets
        )
        
        # CTC Decoder for evaluation
        self.decoder = CTCDecoder(vocab)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        max_grad_norm: float = 5.0,
        log_interval: int = 10
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            max_grad_norm: Maximum gradient norm for clipping
            log_interval: How often to print progress (in batches)
        
        Returns:
            Tuple of (average_loss, cer, wer)
        """
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        start_time = time.time()
        
        for batch_idx, (images, texts) in enumerate(train_loader):
            # Move images to device
            images = images.to(self.device)
            
            # Forward pass
            log_probs = self.model(images)  # (T, B, C)
            
            # Prepare targets for CTC loss
            targets, target_lengths = self._prepare_targets(texts)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)
            
            # Input lengths (all sequences have same length after CNN)
            batch_size = images.size(0)
            input_lengths = torch.full(
                (batch_size,),
                log_probs.size(0),
                dtype=torch.long,
                device=self.device
            )
            
            # Compute CTC loss
            loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Decode predictions for metric calculation
            with torch.no_grad():
                predictions = self.decoder.greedy_decode(log_probs, input_lengths)
                all_predictions.extend(predictions)
                all_targets.extend(texts)
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                batches_done = batch_idx + 1
                batches_total = len(train_loader)
                
                print(
                    f"Epoch {epoch} [{batches_done}/{batches_total}] "
                    f"Loss: {loss.item():.4f} "
                    f"Time: {elapsed:.1f}s"
                )
                
                # Show sample prediction
                if predictions:
                    print(f"  Sample - Target: '{texts[0]}'")
                    print(f"  Sample - Pred:   '{predictions[0]}'")
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        cer = calculate_cer(all_predictions, all_targets)
        wer = calculate_wer(all_predictions, all_targets)
        
        return avg_loss, cer, wer
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int = None
    ) -> Tuple[float, float, float, List[Tuple[str, str]]]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number (for logging)
        
        Returns:
            Tuple of (average_loss, cer, wer, sample_predictions)
            sample_predictions: List of (target, prediction) tuples
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, texts in val_loader:
                # Move images to device
                images = images.to(self.device)
                
                # Forward pass
                log_probs = self.model(images)  # (T, B, C)
                
                # Prepare targets
                targets, target_lengths = self._prepare_targets(texts)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)
                
                # Input lengths
                batch_size = images.size(0)
                input_lengths = torch.full(
                    (batch_size,),
                    log_probs.size(0),
                    dtype=torch.long,
                    device=self.device
                )
                
                # Compute loss
                loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                total_loss += loss.item()
                
                # Decode predictions
                predictions = self.decoder.greedy_decode(log_probs, input_lengths)
                all_predictions.extend(predictions)
                all_targets.extend(texts)
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        cer = calculate_cer(all_predictions, all_targets)
        wer = calculate_wer(all_predictions, all_targets)
        
        # Sample predictions for inspection (first 5)
        sample_predictions = list(zip(all_targets[:5], all_predictions[:5]))
        
        return avg_loss, cer, wer, sample_predictions
    
    def _prepare_targets(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare text targets for CTC loss.
        
        CTC loss expects:
        - targets: Concatenated target sequences (sum of all lengths)
        - target_lengths: Length of each target sequence
        
        Args:
            texts: List of text strings
        
        Returns:
            Tuple of (targets, target_lengths)
        """
        encoded_targets = []
        target_lengths = []
        
        for text in texts:
            encoded = self.vocab.encode(text)
            encoded_targets.extend(encoded)
            target_lengths.append(len(encoded))
        
        targets = torch.tensor(encoded_targets, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        
        return targets, target_lengths


def create_dataloaders(
    dataset,
    batch_size: int = 8,
    train_split: float = 0.8,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a dataset.
    
    Args:
        dataset: Full dataset
        batch_size: Batch size for training
        train_split: Fraction of data to use for training (rest is validation)
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Drop incomplete batches for consistent sequence lengths
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def print_epoch_summary(
    epoch: int,
    train_loss: float,
    train_cer: float,
    train_wer: float,
    val_loss: float,
    val_cer: float,
    val_wer: float,
    sample_predictions: List[Tuple[str, str]] = None
):
    """
    Print formatted epoch summary.
    
    Args:
        epoch: Epoch number
        train_loss: Training loss
        train_cer: Training CER
        train_wer: Training WER
        val_loss: Validation loss
        val_cer: Validation CER
        val_wer: Validation WER
        sample_predictions: Optional list of (target, prediction) tuples
    """
    print("\n" + "=" * 80)
    print(f"EPOCH {epoch} SUMMARY")
    print("=" * 80)
    
    print("\nTraining Metrics:")
    print(f"  Loss: {train_loss:.4f}")
    print(f"  CER:  {train_cer:.2f}%")
    print(f"  WER:  {train_wer:.2f}%")
    
    print("\nValidation Metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  CER:  {val_cer:.2f}%")
    print(f"  WER:  {val_wer:.2f}%")
    
    if sample_predictions:
        print("\nSample Predictions:")
        for i, (target, pred) in enumerate(sample_predictions, 1):
            print(f"  {i}. Target: '{target}'")
            print(f"     Pred:   '{pred}'")
            if target == pred:
                print("     ✓ Perfect match!")
            print()
    
    print("=" * 80 + "\n")


def test_ocr_trainer():
    """Test OCR trainer with dummy data."""
    print("=" * 70)
    print("OCR Trainer Testing")
    print("=" * 70)
    
    # Create dummy model and vocab
    from .crnn_model import CRNN
    from .vocab_ctc import Vocabulary
    
    vocab = Vocabulary()
    model = CRNN(num_classes=vocab.num_classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nDevice: {device}")
    print(f"Vocabulary size: {vocab.num_classes}")
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 1, 128, 1024).to(device)
    texts = ["hello", "world", "test", "data"]
    
    print(f"\nBatch shape: {images.shape}")
    print(f"Texts: {texts}")
    
    # Create trainer
    trainer = OCRTrainer(model, vocab, device)
    
    # Test forward pass with loss calculation
    print("\nTesting loss calculation...")
    model.train()
    log_probs = model(images)
    print(f"Log probs shape: {log_probs.shape}")
    
    targets, target_lengths = trainer._prepare_targets(texts)
    print(f"Targets shape: {targets.shape}")
    print(f"Target lengths: {target_lengths}")
    
    input_lengths = torch.full(
        (batch_size,),
        log_probs.size(0),
        dtype=torch.long,
        device=device
    )
    
    loss = trainer.ctc_loss(
        log_probs,
        targets.to(device),
        input_lengths,
        target_lengths.to(device)
    )
    print(f"CTC Loss: {loss.item():.4f}")
    
    # Test decoding
    print("\nTesting decoding...")
    predictions = trainer.decoder.greedy_decode(log_probs, input_lengths)
    print("Predictions:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        print(f"  {i+1}. Target: '{text}' | Pred: '{pred}'")
    
    print("\n" + "=" * 70)
    print("✓ OCR Trainer test passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_ocr_trainer()