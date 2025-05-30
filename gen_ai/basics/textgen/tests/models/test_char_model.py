"""Module to test char prediction model."""

import torch
from basics.textgen.src.models.char_model import CharacterLevelModel


def test_char_model_output_shape() -> None:
    """
    Test that the CharacterLevelModel outputs have the correct shape and dtype.
    """

    # GIVEN: Model parameters and input tensor
    vocabulary_size = 10
    sequence_length = 4
    batch_size = 2

    model = CharacterLevelModel(
        vocabulary_size=vocabulary_size,
        sequence_length=sequence_length,
    )
    input_sequences = torch.randint(
        low=0, high=vocabulary_size, size=(batch_size, sequence_length)
    )

    # WHEN: Passing input through the model
    output_logits = model(input_sequences)

    # THEN: Output shape and dtype should match expectations
    assert output_logits.shape == (batch_size, vocabulary_size)
    assert output_logits.dtype == torch.float32
