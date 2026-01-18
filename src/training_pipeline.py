"""
Core training logic that can be called from both CLI and queue runners.

This module contains the actual training implementation that is agnostic
to how it was invoked (CLI vs programmatic).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import os

from .crnn_model import CRNN
from .experiment_config import (
    AdamOptimizerConfig,
    CRNNConfig,
    ExperimentConfig,
    ReduceOnPlateauConfig,
)
from .experiment_tracker import ExperimentTracker
from .line_dataset import LineImageTextDataset
from .data_augmentation import get_augmentor
from .ocr_trainer import OCRTrainer
from .vocab_ctc import Vocabulary


class TrainingPipeline:
    """Static class containing all training functions."""

    @staticmethod
    def set_seed(seed: int):
        """Set random seeds for reproducibility. Negative seed for non-deterministic"""
        deterministic = seed >= 0
        if not deterministic:
            return

        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Force cuDNN to be deterministic
        torch.backends.cudnn.deterministic = True
        # Disable the cuDNN auto-tuner (which searches for fastest algorithms)
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def create_data_loaders(
        config: ExperimentConfig,
    ) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
        """Create train/val/test data loaders from config.

        Returns:
            (train_loader, val_loader, test_loader, charset)
        """
        # Create augmentor if enabled
        transform = None
        aug_config = config.augmentation
        if aug_config:
            transform = get_augmentor(aug_config)

        # Load full dataset
        full_dataset = LineImageTextDataset(
            transform=transform,
        )

        # Build character set from all text labels
        all_chars = set()
        for _, text in full_dataset.data_pairs:
            all_chars.update(text)
        charset = sorted(list(all_chars))

        # print(f"Character set size: {len(charset)}")
        # print(f"Characters: {''.join(charset)}")

        # Split dataset
        total_size = len(full_dataset)
        train_size = int(config.data.train_split * total_size)
        val_size = int(config.data.val_split * total_size)
        test_size = total_size - train_size - val_size

        generator = torch.Generator()
        if config.seed >= 0:
            generator.manual_seed(config.seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=generator,
        )

        # print(f"Train size: {len(train_dataset)}")
        # print(f"Val size: {len(val_dataset)}")
        # print(f"Test size: {len(test_dataset)}")

        num_cpu_cores = os.cpu_count()
        num_cpu_cores = num_cpu_cores if num_cpu_cores else 4
        num_workers = num_cpu_cores - 1

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader, charset

    @staticmethod
    def create_model(config: ExperimentConfig, num_classes: int):
        """Create model from config.

        Args:
            config: Experiment configuration
            num_classes: Number of output classes (charset size + blank)

        Returns:
            model: PyTorch model
        """

        if isinstance(config.model, CRNNConfig):
            model_config: CRNNConfig = config.model
            model = CRNN(
                num_classes=num_classes,
                lstm_hidden_size=model_config.lstm_hidden_size,
                lstm_num_layers=model_config.lstm_num_layers,
            )
        else:
            raise ValueError(f"Unknown model type: {type(config.model)}")

        return model

    @staticmethod
    def create_optimizer(model: nn.Module, config: ExperimentConfig):
        """Create optimizer from config."""

        if isinstance(config.optimizer, AdamOptimizerConfig):
            optimizer_config: AdamOptimizerConfig = config.optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {type(config.model)}")

        return optimizer

    @staticmethod
    def create_scheduler(optimizer, config: ExperimentConfig):
        """Create learning rate scheduler from config."""

        if isinstance(config.learning_rate_scheduler, ReduceOnPlateauConfig):
            lrs_config: ReduceOnPlateauConfig = config.learning_rate_scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=lrs_config.factor,
                patience=lrs_config.patience,
            )
        # elif config.training.lr_scheduler == "cosine":
        #     scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #         optimizer, T_max=config.training.epochs
        #     )
        # elif config.training.lr_scheduler == "step":
        #     scheduler = optim.lr_scheduler.StepLR(
        #         optimizer, step_size=10, gamma=config.training.lr_factor
        #     )
        else:
            scheduler = None

        return scheduler

    @staticmethod
    def train_experiment(
        config: ExperimentConfig, tracker_base_dir: str = "_experiments"
    ) -> dict:
        """Main training function that runs a complete experiment.

        Args:
            config: Experiment configuration
            tracker_base_dir: Base directory for experiment tracking

        Returns:
            results: Dictionary with training results
        """
        # Set random seed
        TrainingPipeline.set_seed(config.seed)

        # Create experiment tracker
        tracker = ExperimentTracker(
            base_dir=tracker_base_dir, run_dir=config.experiment_name, unique_folder=False
        )
        tracker.create_run(config.to_dict())

        # Save configuration
        config.save(str(Path(tracker.run_dir) / "config.json"))

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Create data loaders
        print("\nCreating data loaders...")
        train_loader, val_loader, test_loader, charset = (
            TrainingPipeline.create_data_loaders(config)
        )

        # Create model
        print("\nCreating model...")
        num_classes = len(charset) + 1  # +1 for CTC blank
        model = TrainingPipeline.create_model(config, num_classes)
        model = model.to(device)

        # Vocab
        vocab = Vocabulary(custom_chars="".join(charset))

        # Save model architecture
        tracker.save_architecture(model)

        # Create optimizer and scheduler
        optimizer = TrainingPipeline.create_optimizer(model, config)
        scheduler = TrainingPipeline.create_scheduler(optimizer, config)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # TODO: should be configurable
        trainer = OCRTrainer(model, vocab, device)

        # Training loop
        print(f"\nStarting training {config.experiment_name}\n")

        best_val_cer = float("inf")
        best_val_wer = float("inf")
        patience_counter = 0

        epochs_to_train = config.training.epochs + 1
        for epoch in range(1, epochs_to_train):
            # Train
            train_loss, train_cer, train_wer = trainer.train_epoch(
                train_loader, optimizer, epoch, config.training.gradient_clip
            )

            # Validate
            val_loss, val_cer, val_wer, _ = trainer.validate(val_loader, epoch)

            # Log metrics
            tracker.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=100.0 - val_cer,  # Using CER as accuracy proxy
                val_loss=val_loss,
                val_acc=100.0 - val_cer,
            )

            # Update learning rate
            if scheduler is not None:
                if config.learning_rate_scheduler is ReduceOnPlateauConfig:
                    scheduler.step(val_loss)

            # Save best model
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_val_wer = val_wer
                tracker.save_model(model, "best_model.pt")
                patience_counter = 0
                print(f"→ New best model! CER: {val_cer:.4f}, WER: {val_wer:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

            # Print epoch summary
            print(
                f"Epoch {epoch}/{config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}"
            )

        # Save final checkpoint
        tracker.save_checkpoint(
            model, optimizer, epochs_to_train, "final_checkpoint.pt"
        )

        # Test on test set
        print("\nEvaluating on test set...")
        test_loss, test_cer, test_wer, _ = trainer.validate(test_loader)
        print(f"Test CER: {test_cer:.4f}, Test WER: {test_wer:.4f}")

        # Save metrics and plots
        tracker.plot_all()

        # Return results
        results = {
            "best_val_cer": best_val_cer,
            "best_val_wer": best_val_wer,
            "test_loss": test_loss,
            "test_cer": test_cer,
            "test_wer": test_wer,
            "epochs_trained": epochs_to_train,
        }

        # Save final results dictionary for easy access and reproducibility
        tracker.save_results(results)

        return results
