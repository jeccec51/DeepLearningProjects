"""The main module to train."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.generic_model import get_model
from utils.data_loader import get_data_loaders
from utils.visualizations import visualize_feature_maps, visualize_attention_maps

def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int, device: torch.device):
    """Train the model.

    Args:
        model: The model to train.
        train_loader: Data loader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of epochs to train for.
        device: Device to use for training (CPU or GPU).
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
                running_loss = 0.0
    print('Finished Training')


def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate the model on the test set.

    Args:
        model: The model to evaluate.
        test_loader: Data loader for the test data.
        device: Device to use for evaluation (CPU or GPU).

    Returns:
        Accuracy of the model on the test set.
    """

    model.to(device)
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    accuracy = 100 * correct_predictions / total_predictions
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy

@hydra.main(config_path="config", config_name="conf")
def main(config: DictConfig):
    """Main function to run the training, evaluation, and visualization.

    Args:
        config: Configuration dictionary.
    """

    print(OmegaConf.to_yaml(config))

    # Get data loaders
    train_loader, test_loader = get_data_loaders(config.training.dataset, config.model.img_size, config.training.batch_size, config.training.use_yuv)

    # Get model
    model = get_model(config)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Determine device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    print("Training the model...")
    train(model, train_loader, criterion, optimizer, config.training.epochs, device)

    # Evaluate the model
    evaluate(model, test_loader, device)

    # Visualize the model
    print("Visualizing the model...")
    if config.model_type == 'cnn':
        visualize_feature_maps(model, test_loader)
    elif config.model_type == 'vit':
        visualize_attention_maps(model, test_loader)

if __name__ == "__main__":
    main()
