"""Test module for char_encoder."""

from pathlib import Path
import pytest

from basics.src.utilities.char_encoder import (
    build_vocab,
    encode,
    decode,
    CharTockenizer,
)


@pytest.mark.parametrize(
    "text, expected_vocab, expected_ivocab",
    [
        ("abc", {"a": 0, "b": 1, "c": 2}, {0: "a", 1: "b", 2: "c"}),
        ("cab", {"a": 0, "b": 1, "c": 2}, {0: "a", 1: "b", 2: "c"}),
        ("banana", {"a": 0, "b": 1, "n": 2}, {0: "a", 1: "b", 2: "n"}),
        ("", {}, {}),
    ],
)
def test_build_vocab(
    text: str, expected_vocab: dict[str, int], expected_ivocab: dict[str, int]
) -> None:
    """Test module for build vocab."""

    # GIVEN a set of texts
    # WHEN build vocab is called
    vocab, ivocab = build_vocab(text)
    # THEN the returned dicts should match
    assert vocab == expected_vocab
    assert ivocab == expected_ivocab


@pytest.mark.parametrize(
    "text, expected",
    [
        ("abc", [0, 1, 2]),
        ("cab", [2, 0, 1]),
        ("banana", [1, 0, 2, 0, 2, 0]),
        ("", []),
    ],
)
def test_encode(text: str, expected: list[int]) -> None:
    """Test module for encoder."""

    # GIVEN a input text
    # GIVEN a char vocab
    vocab, _ = build_vocab(text)
    # WHEN the encode is called
    # THEN the result should match the expected
    assert encode(text=text, vocab=vocab) == expected

@pytest.mark.parametrize(
    "indices, expected",
    [
        ([0, 1, 2], "abc",),
        ([2, 0, 1], "cab",),
        ([1, 0, 2, 0, 2, 0], "banana",),
        ([], "")
    ]
)
def test_decode(indices: list[int], expected: str) -> None:
    """Test module for decoder"""

    # GIVEN a list of int
    # GIVEN an inverse vocabulary
    _, ivocab = build_vocab(string=expected)
    text = decode(indices=indices, ivocab=ivocab)
    assert text == expected


def test_encode_decode_roundtrip(tokenizer: CharTockenizer) -> None:
    """Module to test class encode . """

    # GIVEN dummy text
    text = "hello"
    # WHEN encoded 
    encoded = tokenizer.encode(text)
    # WHEN decoded the encoded
    decoded = tokenizer.decode(encoded)
    # THEN
    assert decoded == text


def test_save_and_load_tokenizer(tmp_path: Path, sample_corpus: str) -> None:
    """Module to test save nd load."""

    # GIVEN a temp path
    filepath = tmp_path / "vocab.json"

    # WHEN the vocab is sved
    original_tokenizer = CharTockenizer(sample_corpus)
    original_tokenizer.save(str(filepath))

    # WHEN the saved vocab is laoded
    loaded_tokenizer = CharTockenizer.load(str(filepath))

    # Should behave the same
    assert loaded_tokenizer.encode("hello") == \
        original_tokenizer.encode("hello")
    assert loaded_tokenizer.decode([0, 1]) == original_tokenizer.decode([0, 1])
