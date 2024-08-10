import pytest
import numpy as np
from nlp.sentiment_analysis.model import SentimentModel

@pytest.fixture
def sentiment_model():
    return SentimentModel()

def test_model_prediction(sentiment_model):
    text = "The stock market is booming today."
    prediction = sentiment_model.predict(text)
    assert prediction.shape == (1, 2)  # Binary classification
    assert np.isclose(np.sum(prediction), 1.0)  # Probabilities should sum to 1

def test_model_initialization_with_invalid_name():
    with pytest.raises(OSError):
        SentimentModel("invalid_model_name")