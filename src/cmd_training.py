"""
Single experiment training script with CLI interface.

This script maintains the original command-line interface for running
individual training experiments. It wraps the core training logic.

Usage:
    python cmd_training.py --epochs 50 --lr 0.001 --batch-size 16
"""

import argparse

from .training_pipeline import TrainingPipeline
from .experiment_config import (
    AdamOptimizerConfig,
    ExperimentConfig,
    TrainingConfig,
    DataConfig,
)
from .data_augmentation import AugmentationConfig


def parse_args() -> ExperimentConfig:
    """Parse command-line arguments and return ExperimentConfig."""
    parser = argparse.ArgumentParser(
        description="Train handwriting OCR model (single experiment)"
    )

    # Model architecture
    parser.add_argument(
        "--model-type",
        type=str,
        default="crnn",
        choices=["crnn", "transformer", "vit"],
        help="Model architecture type",
    )
    parser.add_argument(
        "--lstm-hidden-size", type=int, default=256, help="LSTM hidden size"
    )
    parser.add_argument(
        "--lstm-num-layers", type=int, default=2, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.25, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="reduce_on_plateau",
        choices=["reduce_on_plateau", "cosine", "step"],
        help="Learning rate scheduler",
    )

    # Data
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Directory containing text files"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="./_inter_data",
        help="Directory containing preprocessed images",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Train split ratio"
    )

    # Augmentation
    parser.add_argument(
        "--no-augmentation", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--aug-rotation-chance",
        type=float,
        default=0.5,
        help="Probability of applying rotation",
    )
    parser.add_argument(
        "--aug-rotation-max-angle",
        type=float,
        default=3.5,
        help="Maximum rotation angle",
    )

    # Experiment metadata
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--description", type=str, default=None, help="Experiment description"
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Config file (alternative to CLI args)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides CLI args)",
    )

    args = parser.parse_args()

    # If config file provided, load it
    if args.config:
        config = ExperimentConfig.load(args.config)
        print(f"Loaded configuration from: {args.config}")
        return config

    # Otherwise, create config from CLI args
    config = ExperimentConfig(
        training=TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
        ),
        optimizer=AdamOptimizerConfig(
            learning_rate=args.lr, weight_decay=args.weight_decay
        ),
        data=DataConfig(train_split=args.train_split),
        augmentation=AugmentationConfig(
            rotation_chance=0.0 if args.no_augmentation else args.aug_rotation_chance,
            rotation_max_angle=args.aug_rotation_max_angle,
        ),
        experiment_name=args.name,
        run_description=args.description,
        seed=args.seed,
    )

    return config


def main():
    """Main training function."""
    # Parse arguments
    config = parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(config)
    print("=" * 60 + "\n")

    # Run training
    results = TrainingPipeline.train_experiment(config)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation CER: {results.get('best_val_cer', 'N/A'):.4f}")
    print(f"Best validation WER: {results.get('best_val_wer', 'N/A'):.4f}")
    print(f"Run directory: {results.get('run_dir', 'N/A')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
