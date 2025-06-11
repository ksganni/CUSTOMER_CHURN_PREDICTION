from src.data_preprocessing import load_data


def test_load_data():
    df = load_data()
    assert df.shape[0] > 0
    assert 'Churn' in df.columns
