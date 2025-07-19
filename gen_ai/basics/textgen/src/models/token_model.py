"""A token level prediction model using a simple MLP."""


import torch
from torch import nn

class TokenLevelModel(nn.Module):
    """Simple model for next token prediction."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_size: int = 32,
    ) -> None:
        """Initialization routine.

        Args:
            vocabulary_size: Total number of unique tokens.
            embedding_size: Size of token embeddings.
        """

        super().__init__()
        self.vocabulary_size = vocabulary_size 
        self.token_embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_size, embedding_dim=embedding_size
        )
        self.linear = nn.Linear(
            in_features=embedding_size,
            out_features=vocabulary_size
        )

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP model.

        Args:
            input_tokens: input tensor (batch_size, sequence_length)

        Returns:
            logits of shape (batch_size, vocabulary_size)
        """

        embedded_tokens = self.token_embedding_layer(input_tokens)
        logits = self.linear(embedded_tokens)
        return logits