"""Fixture for testing the genai modules."""

import pytest
from basics.src.utilities.char_encoder import CharTockenizer


@pytest.fixture
def sample_corpus() -> str:
    """Fixture to return a sample text."""

    return "helo world"


@pytest.fixture
# pylint: disable=redefined-outer-name
def tokenizer(sample_corpus) -> CharTockenizer:
    """Returns a tokenizer class instance."""

    return CharTockenizer(sample_corpus)
