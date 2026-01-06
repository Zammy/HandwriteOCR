"""
Complete OCR Training Script
=============================

This script brings together all components to train a CRNN model
for handwriting OCR with CTC loss.

Usage:
    python train_ocr.py --epochs 50 --batch_size 16 --lr 0.001

The script will:
1. Load your preprocessed line images and labels
2. Create train/val split
3. Initialize CRNN model
4. Train with CTC loss
5. Save checkpoints and metrics
6. Generate plots
"""

import torch
import torch.optim as optim
from pathlib import Path
import argparse

from src.crnn_model import CRNN, count_parameters
from src.vocab_ctc import Vocabulary, calculate_cer, calculate_wer
from src.ocr_trainer import (
    OCRTrainer,
    create_dataloaders,
    print_epoch_summary
)
from src.line_dataset import LineImageTextDataset
from src.data_augmentation import AugmentationConfig, get_augmentor
from src.experiment_tracker import ExperimentTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CRNN OCR model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing text files')
    parser.add_argument('--images_dir', type=str, default='./_inter_data',
                        help='Directory containing line images')
    
    # Model parameters
    parser.add_argument('--lstm_hidden', type=int, default=256,
                        help='LSTM hidden size')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of BiLSTM layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--max_grad_norm', type=float, default=5.0,
                        help='Maximum gradient norm for clipping')
    
    # Augmentation parameters
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--rotation_chance', type=float, default=0.5,
                        help='Probability of applying rotation')
    parser.add_argument('--rotation_max', type=float, default=3.5,
                        help='Maximum rotation angle in degrees')
    parser.add_argument('--shear_chance', type=float, default=0.5,
                        help='Probability of applying shear')
    parser.add_argument('--shear_max', type=float, default=0.2,
                        help='Maximum shear factor')
    parser.add_argument('--elastic_chance', type=float, default=0.5,
                        help='Probability of applying elastic distortion')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log during training')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='How often to save checkpoints (epochs)')
    
    return parser.parse_args()


def setup_augmentation(args):
    """Create augmentation configuration."""
    if args.no_augmentation:
        return None
    
    aug_config = AugmentationConfig(
        rotation_chance=args.rotation_chance,
        rotation_max_angle=args.rotation_max,
        shear_chance=args.shear_chance,
        shear_max_factor=args.shear_max,
        elastic_chance=args.elastic_chance,
    )
    
    return get_augmentor(aug_config)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Print configuration
    print("=" * 80)
    print("CRNN OCR Training Configuration")
    print("=" * 80)
    print(f"\nData:")
    print(f"  Text files:   {args.data_dir}")
    print(f"  Images:       {args.images_dir}")
    print(f"  Train split:  {args.train_split * 100:.0f}%")
    print(f"\nModel:")
    print(f"  LSTM hidden:  {args.lstm_hidden}")
    print(f"  LSTM layers:  {args.lstm_layers}")
    print(f"\nTraining:")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max grad norm: {args.max_grad_norm}")
    print(f"\nAugmentation: {'Disabled' if args.no_augmentation else 'Enabled'}")
    if not args.no_augmentation:
        print(f"  Rotation:     {args.rotation_chance * 100:.0f}% chance, ±{args.rotation_max}°")
        print(f"  Shear:        {args.shear_chance * 100:.0f}% chance, ±{args.shear_max}")
        print(f"  Elastic:      {args.elastic_chance * 100:.0f}% chance")
    print("=" * 80 + "\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab = Vocabulary()
    print(f"Vocabulary: {vocab}\n")
    
    # Load dataset
    print("Loading dataset...")
    transform = setup_augmentation(args)
    dataset = LineImageTextDataset(
        data_dir=args.data_dir,
        images_dir=args.images_dir,
        transform=transform
    )
    
    if len(dataset) == 0:
        print("ERROR: No data found! Please run preprocessing first.")
        print("Run: python -m src.data_preprocess")
        return
    
    print(f"Total samples: {len(dataset)}\n")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = CRNN(
        num_classes=vocab.num_classes,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Expected sequence length: {model.get_sequence_length()}\n")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler (optional but recommended)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        # verbose=True
    )
    
    # Initialize trainer
    trainer = OCRTrainer(model, vocab, device)
    
    # Initialize experiment tracker
    config = vars(args)
    config['num_parameters'] = count_parameters(model)
    config['vocab_size'] = vocab.num_classes
    
    tracker = ExperimentTracker()
    tracker.create_run(config)
    tracker.save_architecture(model)
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    
    best_val_cer = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_cer, train_wer = trainer.train_epoch(
            train_loader,
            optimizer,
            epoch,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval
        )
        
        # Validate
        val_loss, val_cer, val_wer, sample_preds = trainer.validate(
            val_loader,
            epoch
        )
        
        # Update learning rate
        scheduler.step(val_cer)
        
        # Log metrics
        tracker.log_epoch(
            epoch,
            train_loss,
            100 - train_cer,  # Convert CER to "accuracy" for plotting
            val_loss,
            100 - val_cer
        )
        
        # Print summary
        print_epoch_summary(
            epoch,
            train_loss,
            train_cer,
            train_wer,
            val_loss,
            val_cer,
            val_wer,
            sample_preds
        )
        
        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            tracker.save_model(model, 'best_model.pt')
            print(f"✓ New best model! CER: {val_cer:.2f}%\n")
        
        # Save periodic checkpoints
        # if epoch % args.save_interval == 0:
        #     tracker.save_checkpoint(
        #         model,
        #         optimizer,
        #         epoch,
        #         f'checkpoint_epoch_{epoch}.pt'
        #     )
    
    # Save final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nBest validation CER: {best_val_cer:.2f}%")
    print(f"Results saved to: {tracker.run_dir}\n")
    
    # Save final checkpoint and metrics
    tracker.save_checkpoint(model, optimizer, args.epochs, 'final_checkpoint.pt')
    tracker.save_metrics()
    
    # Generate plots
    print("Generating plots...")
    tracker.plot_all()
    tracker.plot_cer_curve('val_acc')  # Note: tracker stores as accuracy
    
    print("\n✓ All files saved successfully!")


if __name__ == '__main__':
    main()