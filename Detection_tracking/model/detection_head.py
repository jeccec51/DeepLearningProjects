"""Detection Head."""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int):
        """Detection head for predicting bounding boxes and class probabilities.
        
        Args:
            hidden_size: Size of the hidden state in the preceding LSTM.
            num_classes: Number of object classes.
        """

        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes + 4)  # Classes + bounding box coordinates
    
    
    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """Forward pass through the detection head.
        
        Args:
            lstm_output: Input tensor from the LSTM of shape (batch_size, hidden_size)
        
        Returns:
            Output tensor with class probabilities and bounding box coordinates
        """

        intermediate_output = torch.relu(self.fc1(lstm_output))
        output_tensor = self.fc2(intermediate_output)
        return output_tensor


if __name__ == "__main__":
    detection_head = DetectionHead(hidden_size=512, num_classes=80)
    print(detection_head)
