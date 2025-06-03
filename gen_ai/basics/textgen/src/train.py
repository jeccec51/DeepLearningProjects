"""Char gen training module."""

from typing import Callable, Optional
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from conf.training_config import TrainConfig
from data.char_dataset import CharDataset
from models.char_model import CharacterLevelModel
from utilities.char_encoder import CharTockenizer


def load_data(
    file_path: str,
    context_window: int,
    batch_size: int,
    train_fraction: float = 0.9,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Loads text data from a file, tokenizes it, and creates train and
    validation DataLoaders.

    Args:
        file_path: Path to the input text file.
        context_window: Length of each input sequence for the model.
        batch_size: Number of samples per batch.
        train_fraction: Fraction of data to use for training (default: 0.9).

    Returns:
        A tuple containing the training DataLoader, validation DataLoader,
        and vocabulary size.
    """

    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()
    tokenizer = CharTockenizer(text=raw_text)
    encoded_text = tokenizer.encode(text=raw_text)
    dataset = CharDataset(data=encoded_text, block_size=context_window)
    train_size = int(train_fraction * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(
        dataset=dataset,
        lengths=[train_size, val_size],
    )
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size)
    return train_loader, val_loader, tokenizer.vocab_size


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    on_step_end: Optional[Callable] = None,
) -> float:
    """Trains the model for one epoch.

    Args:
        model: The neural network model to train.
        loader: DataLoader providing batches of input and target data.
        optimizer: Optimizer for updating model parameters.
        loss_fn: Loss function to compute the training loss.
        device: Device on which to perform computations (CPU or CUDA).
        on_step_end: Optional callback called at the end of each step.

    Returns:
        The average loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    loop = tqdm(enumerate(loader), total=len(loader), desc="Train")

    for step, (input_batch, target_batch) in loop:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = model(input_batch)
        logits = logits.view(-1, logits.size(-1))
        target_batch = target_batch.view(-1)
        loss = loss_fn(logits, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())  # type: ignore

        if on_step_end:
            on_step_end(step, logits, target_batch, loss)
    return total_loss / len(loader)


def train(cfg: TrainConfig) -> None:
    """Trains a text generation model on the provided text data.

    Args:
        cfg: Configuration object containing all training parameters
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data load
    train_data_loader, val_data_loader, vocab_size = load_data(
        file_path=cfg.text_path,
        context_window=cfg.context_window,
        batch_size=cfg.batch_size,
        train_fraction=cfg.training_fraction,
    )
    # Model
    model = CharacterLevelModel(
        vocabulary_size=vocab_size, embedding_size=cfg.embedding_size
    ).to(device)
    # Loss
    loss_function = nn.CrossEntropyLoss()

    # Optimizer selection from config
    optimizer: torch.optim.Optimizer
    if cfg.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    best_val_loss = float('inf')
    check_point_path = cfg.checkpont_path
    # Training loop
    for epoch in range(cfg.num_epochs):
        average_loss = train_one_epoch(
            model=model,
            loader=train_data_loader,
            optimizer=optimizer,
            loss_fn=loss_function,
            device=device,
        )
        val_loss = (
            train_evaluate(
                model=model,
                loader=val_data_loader,
                loss_fn=loss_function,
                device=device,
            )
            if val_data_loader
            else None
        )
        best_val_loss = save_model(
            checkpoint_path=check_point_path,
            current_val_loss=val_loss if val_loss else float('inf'),
            best_val_loss_so_far=best_val_loss,
            model=model,
            current_epoch=epoch,
        )
        print(
            (
                f"Epoch {epoch} | Train Loss: {average_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            if val_loss
            else f"Epoch {epoch} | Train Loss: {average_loss:.4f}"
        )


def train_evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluates the model on the provided data loader.

    Args:
        model: The neural network model to evaluate.
        loader: DataLoader providing batches of input and target data.
        loss_fn: Loss function to compute the evaluation loss.
        device: Device on which to perform computations (CPU or CUDA).

    Returns:
        The average loss over the evaluation dataset.
    """

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs).view(-1, model.vocabulary_size)
            targets = targets.view(-1)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def save_model(
    checkpoint_path: str,
    best_val_loss_so_far: float,
    model: torch.nn.Module,
    current_epoch: int,
    current_val_loss: float,
) -> float:
    """
    Saves the model checkpoint if the current validation loss is greater
    than the best validation loss so far.

    Args:
        checkpoint_path (str): The file path where the model checkpoint will be saved.
        current_val_loss (float): The validation loss from the current epoch.
        best_val_loss_so_far (float): The best validation loss observed so far.
        model (torch.nn.Module): The model to be saved.
        current_epoch (int): The current epoch number.

    Returns:
        float: The updated best validation loss.

    Note:
       The model is saved only if the current validation loss is greater than
       best validation loss.
    """

    if current_val_loss < best_val_loss_so_far:
        best_val_loss_so_far = current_val_loss
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(
            f"Model checkpoint saved at epoch {current_epoch} "
            f"with val loss: {current_val_loss:.4f}"
        )
    return best_val_loss_so_far


if __name__ == "__main__":
    config = TrainConfig()
    train(config)
