import pytest
from nlp.sentiment_analysis.preprocessor import preprocess, clean_text, tokenize, remove_stopwords

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"

def test_tokenize():
    assert tokenize("Hello world") == ["hello", "world"]

def test_remove_stopwords():
    tokens = ["this", "is", "a", "test", "sentence"]
    assert remove_stopwords(tokens) == ["test", "sentence"]

def test_preprocess():
    assert preprocess("This is a test sentence!") == ["test", "sentence"]

@pytest.mark.parametrize("input_text,expected_output", [
    ("Hello, World!", ["hello", "world"]),
    ("This is a test.", ["test"]),
    ("The quick brown fox.", ["quick", "brown", "fox"])
])
def test_preprocess_parametrized(input_text, expected_output):
    assert preprocess(input_text) == expected_output