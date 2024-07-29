"""The main module to train."""
import hydra
import os
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from models.generic_model import get_model as get_pytorch_model
from utils.data_loader import get_data_loaders
from utils.visualizations import visualize_feature_maps, visualize_attention_maps


def train_pytorch(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, 
                  optimizer: optim.Optimizer, epochs: int, device: torch.device, writer: SummaryWriter, metrics: list, log_interval: int) -> None:
    """Train the PyTorch model.

    Args:
        model: The model to train.
        train_loader: Data loader for the training data.
        val_loader: Data loader for the validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of epochs to train for.
        device: Device to use for training (CPU or GPU).
        writer: TensorBoard SummaryWriter for logging.
        metrics: List of metrics to evaluate.
        log_interval: Interval at which to log training and validation loss.
    """

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % log_interval == log_interval - 1:
                avg_train_loss = running_loss / log_interval
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Training Loss: {avg_train_loss:.3f}')
                writer.add_scalar('training_loss', avg_train_loss, epoch * len(train_loader) + batch_idx)
                running_loss = 0.0
                # Evaluate on validation set
                val_loss, val_metrics = evaluate_pytorch(model, val_loader, criterion, device, metrics, epoch * len(train_loader) + batch_idx, prefix="validation")
                print(f'Validation Loss: {val_loss:.3f}')
                writer.add_scalar('validation_loss', val_loss, epoch * len(train_loader) + batch_idx)
                for metric, value in val_metrics.items():
                    writer.add_scalar(f'validation_{metric}', value, epoch * len(train_loader) + batch_idx)

    print('Finished Training')


def evaluate_pytorch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, 
                     metrics: list, step: int, prefix: str = "test") -> tuple:
    """Evaluate the PyTorch model on the test/validation set.

    Args:
        model: The model to evaluate.
        loader: Data loader for the test/validation data.
        criterion: Loss function.
        device: Device to use for evaluation (CPU or GPU).
        metrics: List of metrics to evaluate.
        step: Current step for logging.
        prefix: Prefix for logging (test or validation).

    Returns:
        Tuple of total loss and dictionary of evaluated metrics.
    """

    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    total_loss /= len(loader)
    results = {}
    if 'accuracy' in metrics:
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        results['accuracy'] = accuracy
    if 'precision' in metrics:
        precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
        results['precision'] = precision
    if 'recall' in metrics:
        recall = recall_score(all_labels, all_predictions, average='macro')
        results['recall'] = recall
    if 'f1_score' in metrics:
        f1 = f1_score(all_labels, all_predictions, average='macro')
        results['f1_score'] = f1
    return total_loss, results


@hydra.main(config_path="config", config_name="conf")
def main(config: DictConfig) -> None:
    """Main function to run the training, evaluation, and visualization.

    Args:
        config: Configuration dictionary.
    """

    print(OmegaConf.to_yaml(config))
    log_dir = os.path.abspath(config.training.log_dir)
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        dataset_name=config.training.dataset, 
        img_size=config.model.img_size, 
        batch_size=config.training.batch_size, 
        use_yuv=config.training.use_yuv, 
        run_type=config.training.run_type, 
        short_run_fraction=config.training.short_run_fraction
    )

    # Determine device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(config.training.log_dir)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config.training.log_dir)

    # Get PyTorch model
    model = get_pytorch_model(config)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Train the model
    print("Training the PyTorch model...")
    train_pytorch(model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer,
                   epochs=config.training.epochs, device=device, writer=writer, metrics=config.metrics, log_interval=10)

    # Evaluate the model on test set
    print("Evaluating the PyTorch model on test set...")
    test_loader = val_loader  # Using val_loader as test_loader for simplicity
    test_loss, test_metrics = evaluate_pytorch(model=model, loader=test_loader, criterion=criterion, device=device, metrics=config.metrics, 
                                               step=config.training.epochs * len(train_loader), prefix="test")

    print(f"Test Evaluation Results: Loss: {test_loss:.3f}, Metrics: {test_metrics}")

    # Visualize the model
    print("Visualizing the model...")
    if config.model.name == 'cnn' or config.model.name == 'resnet':
        visualize_feature_maps(model, test_loader)
    elif config.model.name == 'vit':
        visualize_attention_maps(model, test_loader)

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
