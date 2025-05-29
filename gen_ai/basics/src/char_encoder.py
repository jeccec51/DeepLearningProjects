"""Module to encode charactors."""


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
