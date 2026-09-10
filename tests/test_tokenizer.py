"""Tokenizer wrapper tests.

Owner: Prazol. Needs the cached bert-base-uncased tokenizer (downloaded during setup).
"""
import pytest

from src.tokenizer.wrapper import TokenizerWrapper, is_continuation

EXPECTED_SPECIAL_IDS = {"[PAD]": 0, "[UNK]": 100, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103}


@pytest.fixture(scope="module")
def wrapper():
    return TokenizerWrapper()


def test_special_token_ids_stable(wrapper):
    assert wrapper.special_token_ids() == EXPECTED_SPECIAL_IDS


def test_known_string_tokenization(wrapper):
    enc = wrapper.encode("Robustness under character noise.")
    assert enc.pieces == ["robust", "##ness", "under", "character", "noise", "."]
    assert enc.input_ids[0] == 101 and enc.input_ids[-1] == 102
    assert enc.n_special == 2


def test_continuation_and_unk_detection(wrapper):
    assert is_continuation("##ness")
    assert not is_continuation("robust")
    enc = wrapper.encode("pneumonoultramicroscopicsilicovolcanoconiosis")
    assert enc.continuation_count >= 1


def test_unk_on_unknown_script(wrapper):
    enc = wrapper.encode("\U0001F600\U0001F601\U0001F602")  # emoji -> [UNK]
    assert enc.unk_count() >= 1


def test_truncation_flag(wrapper):
    short = wrapper.encode("hello world")
    assert short.truncated is False
    long_text = "word " * (wrapper.max_length + 50)
    assert wrapper.encode(long_text).truncated is True


def test_config_summary_shape(wrapper):
    summary = wrapper.config_summary()
    assert summary["vocab_size"] == 30522
    assert summary["tokenizer_class"] == "BertTokenizer"
