"""base class definitions for tokenizer. """

from typing import Protocol
from abc import abstractmethod


class BaseTokenizer(Protocol):
    """Abstract base class for tokenizers."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary used by the tokenizer."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encodes the input text into a list of integer token IDs."""

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode the integer tokens to strings. """
