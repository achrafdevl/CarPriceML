from src.preprocessing import load_raw, preprocess
from pathlib import Path
import pandas as pd

def test_data_loading():
    df = load_raw(Path("data/car-details.csv"))
    df = preprocess(df)
    assert not df.empty
    assert "selling_price_mad" in df.columns
