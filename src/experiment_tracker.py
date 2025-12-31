import torch
import torch.nn as nn
import torch.optim as optim
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    """Simple experiment tracking system for ML experiments"""

    def __init__(self, base_dir: str = "runs") -> None:
        self.base_dir: Path = Path(base_dir)
        self.run_dir: Path | None
        self.metrics: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def create_run(self, config: dict) -> None:
        """Create a new run directory with timestamp"""
        timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.base_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Created run directory: {self.run_dir}")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
    ) -> None:
        """Log metrics for an epoch"""
        self.metrics["train_loss"].append(train_loss)
        self.metrics["train_acc"].append(train_acc)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
        )

    def save_metrics(self) -> None:
        """Save metrics to JSON"""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save_model(self, model: nn.Module, filename: str = "model.pt") -> None:
        """Save model parameters for inference"""
        torch.save(model.state_dict(), self.run_dir / filename)

    def load_model(self, model: nn.Module, filename: str = "model.pt") -> nn.Module:
        """Load model parameters for inference"""
        model.load_state_dict(torch.load(self.run_dir / filename))
        return model

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        filename: str = "checkpoint.pt",
    ) -> None:
        """Save full training checkpoint"""
        checkpoint: dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": self.metrics,
        }
        torch.save(checkpoint, self.run_dir / filename)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        filename: str = "checkpoint.pt",
    ) -> tuple[nn.Module, optim.Optimizer, int, dict[str, list[float]]]:
        """Load training checkpoint to resume training"""
        checkpoint: dict = torch.load(self.run_dir / filename)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch: int = checkpoint["epoch"]
        metrics: dict[str, list[float]] = checkpoint["metrics"]

        self.metrics = metrics

        return model, optimizer, epoch, metrics

    def save_architecture(self, model: nn.Module) -> None:
        """Save model architecture as text"""
        with open(self.run_dir / "model_arch.txt", "w") as f:
            f.write(str(model))

    def plot_loss_curves(self) -> None:
        """Plot training and validation loss curves"""
        if not self.metrics["train_loss"]:
            return

        epochs: list[int] = list(range(1, len(self.metrics["train_loss"]) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs, self.metrics["train_loss"], "b-", label="Train Loss", linewidth=2
        )
        plt.plot(epochs, self.metrics["val_loss"], "r-", label="Val Loss", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.run_dir / "plots" / "loss_curve.png", dpi=150)
        plt.close()

    def plot_accuracy_curves(self) -> None:
        """Plot training and validation accuracy curves"""
        if not self.metrics["train_acc"]:
            return

        epochs: list[int] = list(range(1, len(self.metrics["train_acc"]) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(
            epochs, self.metrics["train_acc"], "b-", label="Train Accuracy", linewidth=2
        )
        plt.plot(
            epochs, self.metrics["val_acc"], "r-", label="Val Accuracy", linewidth=2
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.run_dir / "plots" / "accuracy_curve.png", dpi=150)
        plt.close()

    def plot_cer_curve(self, metric_name: str = "val_acc") -> None:
        """Plot Character Error Rate curve"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return

        cer_values: list[float] = [100.0 - acc for acc in self.metrics[metric_name]]
        epochs: list[int] = list(range(1, len(cer_values) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, cer_values, "r-", linewidth=2, marker="o", markersize=5)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Character Error Rate (%)", fontsize=12)
        plt.title("Character Error Rate Over Time", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.run_dir / "plots" / "cer_curve.png", dpi=150)
        plt.close()

    def plot_all(self) -> None:
        """Generate all standard plots"""
        self.plot_loss_curves()
        self.plot_accuracy_curves()
        self.plot_cer_curve()
