import utils
import helper
import problem_unittests as tests
from string import punctuation
from collections import Counter
from params import data_dir


def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


# tests.test_create_lookup_tables(create_lookup_tables)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    punctuation_lookup = {
        '.': '<PERIOD>',
        ',': '<COMMA>',
        '"': '<QUOTATION_MARK>',
        ';': '<SEMICOLON>',
        '!': '<EXCLAMATION_MARK>',
        '?': '<QUESTION_MARK>',
        '(': '<LEFT_PAREN>',
        ')': '<RIGHT_PAREN>',
        '-': '<DASH>',
        '\n': '<NEW_LINE>',
    }
    return punctuation_lookup


def helper_preprocess():
    helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)


def helper_load_preprocess_settings():
    int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    return int_text, vocab_to_int, int_to_vocab, token_dict
# tests.test_tokenize(token_lookup)
