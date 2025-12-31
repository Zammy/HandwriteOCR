import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.experiment_tracker import ExperimentTracker
from src.trainer import Trainer


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST digit classification

    Architecture:
        Conv1: 1 -> 32 channels, 3x3 kernel
        Conv2: 32 -> 64 channels, 3x3 kernel
        FC1: 9216 -> 128
        FC2: 128 -> 10
    """

    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1: nn.Conv2d = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2: nn.Conv2d = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1: nn.Linear = nn.Linear(64 * 7 * 7, 128)
        self.fc2: nn.Linear = nn.Linear(128, 10)

        # Other layers
        self.pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        self.relu: nn.ReLU = nn.ReLU()
        self.dropout: nn.Dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: 28x28 -> 14x14
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Conv block 2: 14x14 -> 7x7
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # FC layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def main() -> None:
    # Configuration
    config: dict = {
        "batch_size": 64,
        "epochs": 5,
        "lr": 0.001,
        "seed": 42,
    }

    # Set random seed
    torch.manual_seed(config["seed"])

    # Initialize experiment tracker
    tracker: ExperimentTracker = ExperimentTracker()
    tracker.create_run(config)

    # Data transforms
    transform: transforms.Compose = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset from local directory
    train_dataset: datasets.MNIST = datasets.MNIST(
        "./data/MNIST", train=True, download=True, transform=transform
    )
    val_dataset: datasets.MNIST = datasets.MNIST(
        "./data/MNIST", train=False, download=True, transform=transform
    )

    train_loader: DataLoader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader: DataLoader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Initialize model
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: SimpleCNN = SimpleCNN().to(device)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=config["lr"])

    # Save model architecture
    tracker.save_architecture(model)

    # Initialize trainer
    trainer: Trainer = Trainer()

    print(f"\nTraining on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    # Training loop
    best_val_acc: float = 0.0
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = trainer.train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = trainer.validate(model, val_loader, criterion, device)

        tracker.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

        # Save best model for inference
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            tracker.save_model(model, "best_model.pt")

    # Save final results
    tracker.save_checkpoint(model, optimizer, config["epochs"], "final_checkpoint.pt")
    tracker.save_metrics()
    tracker.save_architecture(model)
    tracker.plot_all()

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Results saved to: {tracker.run_dir}")


if __name__ == "__main__":
    main()
