"""Sequence Model module. """

import torch
import torch.nn as nn


class LSTMSequenceModel(nn.Module):
    """LSTM model for sequence modeling.
        
    Args:
        input_size: Size of the input feature vector.
        hidden_size: Size of the hidden state in LSTM.
        num_layers: Number of LSTM layers."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        """Initialization Routene. """

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the LSTM.
        
        Args:
            sequence_tensor: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """

        lstm_output, _ = self.lstm(sequence_tensor)
        return lstm_output


if __name__ == "__main__":
    sequence_model = LSTMSequenceModel(input_size=64*7*7, hidden_size=512, num_layers=2)
    print(sequence_model)
