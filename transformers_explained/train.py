"""The main module to train."""

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from models.generic_model import get_model
from utils.data_loader import get_data_loaders
from utils.visualizations import visualize_feature_maps, visualize_attention_maps


def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int) -> None:
    """Train the model.

    Args:
        model: The model to train.
        train_loader: Data loader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        epochs: Number of epochs to train for.
    """

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    print('Finished Training')


def evaluate(model: nn.Module, test_loader: DataLoader) -> float:
    """ Evaluate the model on the test set.

    Args:
        model: The model to evaluate.
        test_loader: Data loader for the test data.

    Returns:
        float: Accuracy of the model on the test set.
    """

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy

@hydra.main(config_path="config", config_name="conf")
def main(config: DictConfig):
    """Main function to run the training, evaluation, and visualization.

    Args:
        config: Configuration dictionary.
    """

    # Get data loaders
    train_loader, test_loader = get_data_loaders(config.training.dataset, config.model.img_size, config.training.batch_size)

    # Get model
    model = get_model(config)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Train the model
    print("Training the model...")
    train(model, train_loader, criterion, optimizer, config.training.epochs)

    # Evaluate the model
    print("Evaluating the model...")
    evaluate(model, test_loader)

    # Visualize the model
    print("Visualizing the model...")
    if config.model.type == 'cnn':
        visualize_feature_maps(model, test_loader)
    elif config.model.type == 'vit':
        visualize_attention_maps(model, test_loader)

if __name__ == "__main__":
    main()
