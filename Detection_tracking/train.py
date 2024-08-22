"""Training Module."""
import os
import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model.combined_model import ObjectDetectionAndTrackingModel
from data_loader import create_data_loaders
from typing import Tuple


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_cls: torch.nn.Module,
    criterion_reg: torch.nn.Module,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
    writer: SummaryWriter
) -> float:
    """Trains the model for one epoch.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for the training dataset.
        optimizer: The optimizer used for training.
        criterion_cls: The classification loss function.
        criterion_reg: The regression loss function.
        device: The device on which to perform the training (CPU or GPU).
        epoch: The current epoch number.
        cfg: The configuration dictionary.
        writer: The TensorBoard writer for logging metrics.

    Returns:
        The average training loss for this epoch.
    """

    model.train()
    train_loss = 0.0
    correct_preds = 0
    total_preds = 0
    mae = 0.0
    for batch_idx, (frames, annotations) in enumerate(train_loader):
        frames, annotations = frames.to(device), [ann.to(device) for ann in annotations]
        optimizer.zero_grad()
        outputs = model(frames)
        loss_cls = criterion_cls(outputs[:, :cfg.model.num_classes], torch.tensor([ann[0] for ann in annotations], device=device))
        loss_reg = criterion_reg(outputs[:, cfg.model.num_classes:], torch.tensor([ann[1:] for ann in annotations], device=device))
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs[:, :cfg.model.num_classes], 1)
        correct_preds += (predicted == torch.tensor([ann[0] for ann in annotations], device=device)).sum().item()
        total_preds += len(annotations)

        # Calculate mean absolute error (MAE)
        mae += torch.abs(outputs[:, cfg.model.num_classes:] - torch.tensor([ann[1:] for ann in annotations], device=device)).sum().item()

    train_loss /= len(train_loader)
    accuracy = correct_preds / total_preds
    mae /= total_preds

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)
    writer.add_scalar("MAE/train", mae, epoch)

    print(f"Epoch {epoch+1}/{cfg.train.num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {accuracy:.4f}, Train MAE: {mae:.4f}")

    return train_loss


def validate_model(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion_cls: torch.nn.Module,
    criterion_reg: torch.nn.Module,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
    writer: SummaryWriter
) -> Tuple[float, float, float]:
    """Validates the model on the validation dataset.

    Args:
        model: The model to be validated.
        val_loader: DataLoader for the validation dataset.
        criterion_cls: The classification loss function.
        criterion_reg: The regression loss function.
        device: The device on which to perform the validation (CPU or GPU).
        epoch: The current epoch number.
        cfg: The configuration dictionary.
        writer: The TensorBoard writer for logging metrics.

    Returns:
        The average validation loss, validation accuracy, and validation MAE.
    """
    model.eval()
    val_loss = 0.0
    val_correct_preds = 0
    val_total_preds = 0
    val_mae = 0.0
    with torch.no_grad():
        for frames, annotations in val_loader:
            frames, annotations = frames.to(device), [ann.to(device) for ann in annotations]
            outputs = model(frames)
            loss_cls = criterion_cls(outputs[:, :cfg.model.num_classes], torch.tensor([ann[0] for ann in annotations], device=device))
            loss_reg = criterion_reg(outputs[:, cfg.model.num_classes:], torch.tensor([ann[1:] for ann in annotations], device=device))
            loss = loss_cls + loss_reg
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs[:, :cfg.model.num_classes], 1)
            val_correct_preds += (predicted == torch.tensor([ann[0] for ann in annotations], device=device)).sum().item()
            val_total_preds += len(annotations)

            # Calculate mean absolute error (MAE)
            val_mae += torch.abs(outputs[:, cfg.model.num_classes:] - torch.tensor([ann[1:] for ann in annotations], device=device)).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = val_correct_preds / val_total_preds
    val_mae /= val_total_preds

    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_accuracy, epoch)
    writer.add_scalar("MAE/val", val_mae, epoch)

    print(f"Epoch {epoch+1}/{cfg.train.num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation MAE: {val_mae:.4f}")

    return val_loss, val_accuracy, val_mae


def test_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion_cls: torch.nn.Module,
    criterion_reg: torch.nn.Module,
    device: torch.device,
    cfg: DictConfig
) -> Tuple[float, float, float]:
    """Tests the model on the test dataset.

    Args:
        model: The trained model to be tested.
        test_loader: DataLoader for the test dataset.
        criterion_cls: The classification loss function.
        criterion_reg: The regression loss function.
        device: The device on which to perform the testing (CPU or GPU).
        cfg: The configuration dictionary.

    Returns:
        The average test loss, test accuracy, and test MAE.
    """
    model.eval()
    test_loss = 0.0
    test_correct_preds = 0
    test_total_preds = 0
    test_mae = 0.0
    with torch.no_grad():
        for frames, annotations in test_loader:
            frames, annotations = frames.to(device), [ann.to(device) for ann in annotations]
            outputs = model(frames)
            loss_cls = criterion_cls(outputs[:, :cfg.model.num_classes], torch.tensor([ann[0] for ann in annotations], device=device))
            loss_reg = criterion_reg(outputs[:, cfg.model.num_classes:], torch.tensor([ann[1:] for ann in annotations], device=device))
            loss = loss_cls + loss_reg
            test_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs[:, :cfg.model.num_classes], 1)
            test_correct_preds += (predicted == torch.tensor([ann[0] for ann in annotations], device=device)).sum().item()
            test_total_preds += len(annotations)

            # Calculate mean absolute error (MAE)
            test_mae += torch.abs(outputs[:, cfg.model.num_classes:] - torch.tensor([ann[1:] for ann in annotations], device=device)).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = test_correct_preds / test_total_preds
    test_mae /= test_total_preds

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test MAE: {test_mae:.4f}")

    return test_loss, test_accuracy, test_mae


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """Main function for training the object detection and tracking model.
    
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model configuration
    model_cfg = cfg.model
    
    # Load data loader configuration
    data_loader_cfg = cfg.data_loader
    
    # Load training configuration
    train_cfg = cfg.train

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
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            device=device,
            epoch=epoch,
            cfg=cfg,
            writer=writer
        )

        val_loss, val_accuracy, val_mae = validate_model(
            model=model,
            val_loader=val_loader,
            criterion_cls=criterion_cls,
            criterion_reg=criterion_reg,
            device=device,
            epoch=epoch,
            cfg=cfg,
            writer=writer
        )

        # Save checkpoint
        if not os.path.exists(train_cfg.checkpoint_dir):
            os.makedirs(train_cfg.checkpoint_dir)
        torch.save(model.state_dict(), os.path.join(train_cfg.checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

    # Close TensorBoard writer
    writer.close()

    # Testing phase
    test_loss, test_accuracy, test_mae = test_model(
        model=model,
        test_loader=test_loader,
        criterion_cls=criterion_cls,
        criterion_reg=criterion_reg,
        device=device,
        cfg=cfg
    )

if __name__ == "__main__":
    main()

