import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Trainer:
    """Handles training and validation logic"""

    @staticmethod
    def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Train for one epoch

        Returns:
            tuple: (average_loss, accuracy_percentage)
        """
        model.train()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        for batch_idx, (data, target) in enumerate(loader):
            data: torch.Tensor = data.to(device)
            target: torch.Tensor = target.to(device)

            optimizer.zero_grad()
            output: torch.Tensor = model(data)
            loss: torch.Tensor = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred: torch.Tensor = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss: float = total_loss / len(loader)
        accuracy: float = 100.0 * correct / total
        return avg_loss, accuracy

    @staticmethod
    def validate(
        model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
    ) -> tuple[float, float]:
        """Validate the model

        Returns:
            tuple: (average_loss, accuracy_percentage)
        """
        model.eval()
        total_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for data, target in loader:
                data: torch.Tensor = data.to(device)
                target: torch.Tensor = target.to(device)
                output: torch.Tensor = model(data)
                loss: torch.Tensor = criterion(output, target)

                total_loss += loss.item()
                pred: torch.Tensor = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss: float = total_loss / len(loader)
        accuracy: float = 100.0 * correct / total
        return avg_loss, accuracy
