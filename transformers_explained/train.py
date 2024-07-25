"""The main module to train."""
import hydra
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


def train_pytorch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int, device: torch.device, writer: SummaryWriter):
    """
    Train the PyTorch model.

    Args:
        model: The model to train.
        train_loader: Data loader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of epochs to train for.
        device: Device to use for training (CPU or GPU).
        writer: TensorBoard SummaryWriter for logging.
    """

    model.to(device)
    model.train()
    for epoch in range(epochs):
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
            if batch_idx % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 200:.3f}')
                writer.add_scalar('training_loss', running_loss / 200, epoch * len(train_loader) + batch_idx)
                running_loss = 0.0
    print('Finished Training')


def evaluate_pytorch(model: nn.Module, test_loader: DataLoader, device: torch.device, metrics: list, writer: SummaryWriter, epoch: int) -> dict:
    """Evaluate the PyTorch model on the test set.

    Args:
        model: The model to evaluate.
        test_loader: Data loader for the test data.
        device: Device to use for evaluation (CPU or GPU).
        metrics: List of metrics to evaluate.
        writer: TensorBoard SummaryWriter for logging.
        epoch: Current epoch number.

    Returns:
        Dictionary of evaluated metrics.
    """

    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    results = {}
    if 'accuracy' in metrics:
        accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
        results['accuracy'] = accuracy
        writer.add_scalar('accuracy', accuracy, epoch)
    if 'precision' in metrics:
        precision = precision_score(all_labels, all_predictions, average='macro')
        results['precision'] = precision
        writer.add_scalar('precision', precision, epoch)
    if 'recall' in metrics:
        recall = recall_score(all_labels, all_predictions, average='macro')
        results['recall'] = recall
        writer.add_scalar('recall', recall, epoch)
    if 'f1_score' in metrics:
        f1 = f1_score(all_labels, all_predictions, average='macro')
        results['f1_score'] = f1
        writer.add_scalar('f1_score', f1, epoch)
    return results


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    """Main function to run the training, evaluation, and visualization.

    Args:
        config: Configuration dictionary.
    """
    print(OmegaConf.to_yaml(config))

    # Get data loaders
    train_loader, test_loader = get_data_loaders(config.training.dataset, config.model.cnn.img_size, config.training.batch_size, config.training.use_yuv)

    # Determine device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=config.training.log_dir)

    # Get PyTorch model
    model = get_pytorch_model(config)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Train the model
    print("Training the PyTorch model...")
    train_pytorch(model, train_loader, criterion, optimizer, config.training.epochs, device, writer)

    # Evaluate the model
    print("Evaluating the PyTorch model...")
    results = evaluate_pytorch(model, test_loader, device, config.training.metrics, writer, config.training.epochs)

    print(f"Evaluation Results: {results}")

    # Visualize the model
    print("Visualizing the model...")
    if config.model_type == 'cnn':
        visualize_feature_maps(model, test_loader)
    elif config.model_type == 'vit':
        visualize_attention_maps(model, test_loader)

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
