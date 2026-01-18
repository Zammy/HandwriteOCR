from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import json

from .data_augmentation import AugmentationConfig


@dataclass
class ModelConfig:
    dropout: float = 0.25
    trainer: str = ""


@dataclass
class CRNNConfig(ModelConfig):
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    trainer: str = "OCRTrainer"


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 50

    gradient_clip: float = 5.0
    early_stopping_patience: int = 10
    save_best_only: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 0.001


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    weight_decay: float = 1e-5


@dataclass
class LearningRateSchedulerConfig:
    factor: float = 0.5


@dataclass
class ReduceOnPlateauConfig(LearningRateSchedulerConfig):
    patience: int = 5


@dataclass
class DataConfig:
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment.

    All training hyperparameters and settings are defined here.
    """

    model: ModelConfig = field(default_factory=CRNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=AdamOptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    learning_rate_scheduler: LearningRateSchedulerConfig = field(
        default_factory=ReduceOnPlateauConfig
    )

    experiment_name: str = "Experiment"
    run_description: Optional[str] = None

    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def load(cls, filepath: str) -> "ExperimentConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["ExperimentConfig:"]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def get_baseline_config() -> ExperimentConfig:
    """Baseline configuration with minimal augmentation."""
    return ExperimentConfig(
        experiment_name="baseline",
        run_description="Baseline CRNN with minimal augmentation",
        augmentation=AugmentationConfig(
            rotation_chance=0.0,
            shear_chance=0.0,
            thickness_chance=0.0,
            elastic_chance=0.0,
        ),
        training=TrainingConfig(epochs=5),
    )


def get_heavy_augmentation_config() -> ExperimentConfig:
    """Configuration with aggressive augmentation."""
    return ExperimentConfig(
        experiment_name="heavy_aug",
        run_description="Heavy augmentation experiment",
        augmentation=AugmentationConfig(
            rotation_chance=0.8,
            rotation_max_angle=5.0,
            shear_chance=0.8,
            elastic_chance=0.8,
            elastic_alpha=5.0,
        ),
        training=TrainingConfig(epochs=50),
    )


def get_large_model_config() -> ExperimentConfig:
    """Configuration with larger model capacity."""
    return ExperimentConfig(
        experiment_name="large_model",
        run_description="Larger LSTM with more capacity",
        model=CRNNConfig(),
        training=TrainingConfig(batch_size=8, epochs=50),
    )


def get_learning_rate_search_configs() -> list[ExperimentConfig]:
    """Generate multiple configs for learning rate search."""
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    configs = []

    for lr in learning_rates:
        config = ExperimentConfig(
            experiment_name=f"lr_search_{lr}",
            run_description=f"Learning rate search: {lr}",
            training=TrainingConfig(epochs=30),
            optimizer=AdamOptimizerConfig(lr),
        )
        configs.append(config)

    return configs


def get_batch_size_search_configs() -> list[ExperimentConfig]:
    """Generate multiple configs for batch size search."""
    batch_sizes = [4, 8, 16, 32]
    configs = []

    for bs in batch_sizes:
        config = ExperimentConfig(
            experiment_name=f"batch_size_{bs}",
            run_description=f"Batch size search: {bs}",
            training=TrainingConfig(batch_size=bs, epochs=30),
        )
        configs.append(config)

    return configs


if __name__ == "__main__":
    exp = ExperimentConfig()
    exp.save("test_save.json")
