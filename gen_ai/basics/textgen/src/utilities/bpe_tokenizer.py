"""BPE Tokenizer utility using SentencePiece."""

import sentencepiece as spm

from tokenizer_base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Class to tokenize the strings."""

    def __init__(self, model_file: str) -> None:
        """Initialization."""
        self.sentencepiece_processor = spm.SentencePieceProcessor()
        self.sentencepiece_processor.Load(model_file=model_file)
        self.pad_token_id = self.sentencepiece_processor.pad_id()
        self.bos_token_id = self.sentencepiece_processor.bos_id()
        self.eos_token_id = self.sentencepiece_processor.eos_id()
 
    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary used by the tokenizer.

        Returns:
            The number of unique tokens in the tokenizer's vocabulary.
        """
        return self.sentencepiece_processor.vocab_size()
  
    def encode(self, text: str) -> list[int]:
        """
        Encodes the input text into a sequence of integer token IDs using the
        underlying SentencePiece tokenizer.

        Args:
            text: The input text to be tokenized.

        Returns:
            A list of integer token IDs representing the tokenized input text.
        """
        return self.sentencepiece_processor.Encode(text, out_type=int)

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a sequence of token IDs back into a string.

        Args:
            token_ids: The sequence of token IDs to decode.

        Returns:
            str: The decoded string corresponding to the input token IDs.
        """
        return self.sentencepiece_processor.Decode(token_ids)
