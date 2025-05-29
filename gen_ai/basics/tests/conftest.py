"""Fixture for testing the genai modules."""

import pytest
from basics.src.char_encoder import CharTockenizer


@pytest.fixture
def sample_corpus() -> str:
    """Fixture to return a sample text."""

    return "helo world"


@pytest.fixture
def tokenizer(sample_corpus) -> CharTockenizer:  # pylint: disable=redefined-outer-name
    """Returns a tokenizer class instance."""

    return CharTockenizer(sample_corpus)
