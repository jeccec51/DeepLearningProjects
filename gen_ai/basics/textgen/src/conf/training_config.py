"""Config class. """

from dataclasses import dataclass

@dataclass
class TrainConfig:
    """Training config class. """

    text_path: str = "basics/textgen/tests/test_data/input.txt"
    checkpont_path = "basics/textgen/outputs/best.pth"
    context_window: int = 8
    batch_size: int = 4
    num_epochs: int = 1000
    embedding_size: int = 32
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    training_fraction = 0.9

@dataclass
class TokenDataLoaderConfig:
    """Token data loader config class. """

    text_path: str = "basics/textgen/tests/test_data/token/input.txt"
    context_window: int = 8
    batch_size: int = 4
    training_fraction: float = 0.9