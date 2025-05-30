"""Module to test char prediction model. """

import torch
from basics.src.models.char_model import CharacterLevelModel


def test_char_model_output_shape() -> None:
    """Test module to test the char model outputs."""

    vocab_size = 10
    block_size = 4
    batch_size = 2

    model = CharacterLevelModel(
        vocabulary_size=vocab_size,
        sequence_length=block_size,
    )
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, block_size))
    logits = model(x)

    assert logits.shape == (batch_size, vocab_size)
    assert logits.dtype == torch.float32
