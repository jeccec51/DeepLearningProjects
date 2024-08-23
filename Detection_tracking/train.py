"""Training Module."""

import os
import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.combined_model import ObjectDetectionAndTrackingModel
from data_loader.data_loader import create_data_loaders
from typing import Tuple, List


def calculate_metrics(
    outputs: torch.Tensor,
    annotations: List[torch.Tensor],
    num_classes: int,
    device: torch.device
) -> Tuple[float, float]:
    """Calculates accuracy and mean absolute error (MAE).

    Args:
        outputs: The model outputs.
        annotations: The ground truth annotations.
        num_classes: The number of classes in the model.
        device: The device on which the calculations are performed.

    Returns:
        A tuple containing accuracy and mean absolute error.
    """

    # Calculate accuracy
    _, predicted = torch.max(outputs[:, :num_classes], 1)
    correct_preds = (predicted == torch.tensor([ann[0] for ann in annotations], device=device)).sum().item()
    total_preds = len(annotations)

    # Calculate mean absolute error (MAE)
    mae = torch.abs(outputs[:, num_classes:] - torch.tensor([ann[1:] for ann in annotations], device=device)).sum().item()

    accuracy = correct_preds / total_preds
    mae /= total_preds

    return accuracy, mae


def run_model_phase(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion_cls: torch.nn.Module,
    criterion_reg: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    is_training: bool = False,
    optimizer: torch.optim.Optimizer = None
) -> Tuple[float, float, float]:
    """Runs a single phase (training/validation/testing) of the model.

    Args:
        model: The model to be run.
        data_loader: DataLoader for the dataset.
        criterion_cls: The classification loss function.
        criterion_reg: The regression loss function.
        device: The device on which to perform the phase (CPU or GPU).
        num_classes: The number of classes in the model.
        is_training: Whether this is a training phase (default: False).
        optimizer: The optimizer used for training (only needed for training).

    Returns:
        The average loss, accuracy, and MAE for this phase.
    """

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_accuracy = 0.0
    total_mae = 0.0

    for frames, annotations in data_loader:
        frames, annotations = frames.to(device), [ann.to(device) for ann in annotations]

        if is_training:
            optimizer.zero_grad()

        outputs = model(frames)
        loss_cls = criterion_cls(outputs[:, :num_classes], torch.tensor([ann[0] for ann in annotations], device=device))
        loss_reg = criterion_reg(outputs[:, num_classes:], torch.tensor([ann[1:] for ann in annotations], device=device))
        loss = loss_cls + loss_reg

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        accuracy, mae = calculate_metrics(outputs, annotations, num_classes, device)
        total_accuracy += accuracy
        total_mae += mae

    total_loss /= len(data_loader)
    total_accuracy /= len(data_loader)
    total_mae /= len(data_loader)

    return total_loss, total_accuracy, total_mae


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main function for training the object detection and tracking model.
    
    Args:
        cfg: Configuration composed by Hydra.
    """

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configuration
    model_cfg = cfg.model.model
    
    # Load data loader configuration
    data_loader_cfg = cfg.data_loader.data_loader
    
    # Load training configuration
    train_cfg = cfg.train.train

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        video_dir=data_loader_cfg.video_dir,
        annotation_dir=data_loader_cfg.annotation_dir,
        batch_size=data_loader_cfg.batch_size,
        seq_len=data_loader_cfg.seq_len,
        num_workers=data_loader_cfg.num_workers
    )

    # Initialize model
    model = ObjectDetectionAndTrackingModel(
        num_classes=model_cfg.num_classes,
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers
    ).to(device)
    
    # Select optimizer
    if train_cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    elif train_cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=train_cfg.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer}")

    # Select loss functions
    if train_cfg.loss_fn_cls == "cross_entropy":
        criterion_cls = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported classification loss function: {train_cfg.loss_fn_cls}")

    if train_cfg.loss_fn_reg == "smooth_l1":
        criterion_reg = torch.nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported regression loss function: {train_cfg.loss_fn_reg}")

    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "logs"))

    # Training loop across epochs
    for epoch in range(train_cfg.num_epochs):
        train_loss, train_accuracy, train_mae = run_model_phase(
            model=model,
            data_loader=train_loader,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            device=device,
            num_classes=model_cfg.num_classes,
            is_training=True,
            optimizer=optimizer
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("MAE/train", train_mae, epoch)

        val_loss, val_accuracy, val_mae = run_model_phase(
            model=model,
            data_loader=val_loader,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            device=device,
            num_classes=model_cfg.num_classes,
            is_training=False
        )

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)

        print(f"Epoch {epoch+1}/{train_cfg.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train MAE: {train_mae:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation MAE: {val_mae:.4f}")

        # Save checkpoint
        if not os.path.exists(train_cfg.checkpoint_dir):
            os.makedirs(train_cfg.checkpoint_dir)
        torch.save(model.state_dict(), os.path.join(train_cfg.checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

    # Close TensorBoard writer
    writer.close()

    # Testing phase
    test_loss, test_accuracy, test_mae = run_model_phase(
        model=model,
        data_loader=test_loader,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        device=device,
        num_classes=model_cfg.num_classes,
        is_training=False
    )

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test MAE: {test_mae:.4f}")


if __name__ == "__main__":
    main()
