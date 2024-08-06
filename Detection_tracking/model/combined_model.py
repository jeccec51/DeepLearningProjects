"""Detection and tracking model. """

import torch
import torch.nn as nn
from model.efficientnet_backbone import EfficientNetBackbone
from model.bifpn import BiFPN
from model.sequence_model import LSTMSequenceModel
from model.detection_head import DetectionHead


class ObjectDetectionAndTrackingModel(nn.Module):
    """Combined model for object detection and tracking.
        
        Args:
            num_classes: Number of object classes.
            hidden_size: Size of the hidden state in the LSTM.
            num_layers: Number of LSTM layers.
        """

    def __init__(self, num_classes: int, hidden_size: int = 512, num_layers: int = 2):
        """Initialization routene. """

        super(ObjectDetectionAndTrackingModel, self).__init__()
        self.backbone = EfficientNetBackbone(model_name='efficientnet_b0')
        self.bifpn = BiFPN(in_channels=40, out_channels=64)
        self.sequence_model = LSTMSequenceModel(input_size=64*7*7, hidden_size=hidden_size, num_layers=num_layers)
        self.detection_head = DetectionHead(hidden_size, num_classes)
    

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the combined model.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, C, H, W)
        
        Returns:
            Output tensor with class probabilities and bounding box coordinates
        """

        batch_size, seq_len, num_channels, height, width = input_tensor.size()
        input_tensor = input_tensor.view(batch_size * seq_len, num_channels, height, width)
        
        # EfficientNet backbone
        backbone_features = self.backbone(input_tensor)
        
        # BiFPN
        bifpn_features = self.bifpn(backbone_features)
        
        # Flatten BiFPN output and reshape for LSTM
        flattened_bifpn_features = [feature_map.view(batch_size, seq_len, -1) for feature_map in bifpn_features]
        lstm_input = torch.cat(flattened_bifpn_features, dim=-1)
        
        # Sequence Model
        lstm_output = self.sequence_model(lstm_input)
        
        # Detection Head
        output_tensor = self.detection_head(lstm_output[:, -1, :])  # Use the last output of LSTM
        return output_tensor

if __name__ == "__main__":
    model = ObjectDetectionAndTrackingModel(num_classes=80)
    print(model)
