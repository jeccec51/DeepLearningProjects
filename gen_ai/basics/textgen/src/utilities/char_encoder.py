"""Module to encode charactors."""

import json


def build_vocab(string: str) -> tuple[dict[str, int], dict[int, str]]:
    """Function that builds fw and rev dicts of charactor to int encoding.

    Args:
        string: The incoming character stream

    Returns:
        Two dictionaries with character to int, and int to charactor mapping
    """

    unique_charactors = sorted(set(string))
    vocab = {ch: idx for idx, ch in enumerate(unique_charactors)}
    ivocab = {idx: ch for ch, idx in vocab.items()}
    return vocab, ivocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    """Converts incoming string to list of ints.

    Args:
        text: incoming text
        vocab: the char to int matching 

    Returns:
        a list of ints reprasenting the string
    """

    return [vocab[ch] for ch in text]


def decode(indices: list[int], ivocab: dict[int, str]) -> str:
    """Decodes a list of ints to a text.

    Args:
        indices: a list of ints reprasenting the encoded text
        ivocab: the char to int mapping

    Returns:
        a decoded text
    """

    return ''.join([ivocab[index] for index in indices])


class CharTockenizer:
    """Module to tockanize the string."""

    def __init__(self, text: str) -> None:
        """Initialize the module.

        Args:
            text: input text for tockanization
        """

        self._vocab, self._ivocab = build_vocab(string=text)

    @property
    def vocab_size(self) -> int:
        """Returns the size of vocab.

        Returns:
            _vocab length
        """

        return len(self._vocab)

    def encode(self, text: str) -> list[int]:
        """Module to encode the string. 

        Args:
            text: text to encode
        """

        return encode(text=text, vocab=self._vocab)

    def decode(self, indices: list[int]) -> str:
        """Module to decode the encoded list.

        Args:
            indices: List of decoded 

        Returns:
            the decoded string"""

        return decode(indices=indices, ivocab=self._ivocab)

    def save(self, filepath: str) -> None:
        """Module to save the string as json.

        Args:
            filepath: the path to save the string 
        """

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f)

    @classmethod
    def load(cls, filepath: str) -> "CharTockenizer":
        """Clas module to load from file.

        Args:
            filepath: file path from wehre text has to be loaded. 
        """

        with open(filepath, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        obj = cls.__new__(cls)
        obj._vocab = vocab
        obj._ivocab = {idx: ch for ch, idx in vocab.items()}
        return obj
