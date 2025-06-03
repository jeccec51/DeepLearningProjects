"""Module to add accuracy metric call back."""

from typing import Callable

import torch


def accuracy_callback_factory() -> Callable:
    """Factory function to create an accuracy callback for tracking model
    accuracy during training.

    Returns:
       An instance of AccuracyCallback that can be called as a
       function and has a get_final_accuracy method.
    """

    class AccuracyCallback:
        """Callback class for tracking and
        reporting accuracy during model training"""

        def __init__(self) -> None:
            """Initialization. """

            self.num_correct_predictions = 0
            self.num_total_predictions = 0

        def __call__(self, step: int, logits: torch.Tensor,
                     targets: torch.Tensor, loss: float) -> None:
            """Callback to be called at the end of each training step to update
            accuracy statistics.

            Args:
                step: The current training step.
                logits: The model output logits.
                targets: The ground truth labels.
                loss: The loss value for the current step.
            """

            predicted_labels = torch.argmax(logits, dim=-1)
            correct_count = (predicted_labels == targets).sum().item()
            total_count = targets.numel()
            self.num_correct_predictions += int(correct_count)
            self.num_total_predictions += total_count

            if step % 10 == 0:
                accuracy = (
                    self.num_correct_predictions / self.num_total_predictions
                ) * 100
                print(f"    [Step {step}] Accuracy so far: {accuracy:.2f}%")

        def get_final_accuracy(self) -> float:
            """Returns the final computed accuracy.

            Returns:
                float: The final accuracy as a float between 0 and 1.
            """
            if self.num_total_predictions > 0:
                return (
                    self.num_correct_predictions / self.num_total_predictions
                )
            else:
                return 0.0

    return AccuracyCallback()
