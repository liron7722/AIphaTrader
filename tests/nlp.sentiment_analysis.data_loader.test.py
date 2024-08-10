import pytest
import pandas as pd
from nlp.sentiment_analysis.data_loader import load_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'text': ['Positive news', 'Negative news'],
        'sentiment': [1, 0]
    })

def test_load_data(tmp_path, sample_data):
    # Create a temporary CSV file
    csv_file = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_file, index=False)
    
    # Test the load_data function
    texts, labels = load_data(csv_file)
    assert texts == ['Positive news', 'Negative news']
    assert labels == [1, 0]