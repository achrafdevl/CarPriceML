import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
from .config import DATA_PATH

INR_TO_MAD = 0.10  # 1 INR = 0.10 MAD

def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop_duplicates(inplace=True)

    if "selling_price" in df.columns:
        df["selling_price_mad"] = df["selling_price"].astype(float) * INR_TO_MAD
    else:
        raise KeyError("selling_price column not found in the dataset")

    num_cols = ["year", "km_driven", "mileage_mpg", "engine_cc", "max_power_bhp", "torque_nm", "seats"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            raise KeyError(f"Column {c} not found in the dataset")

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    keep = ["company", "model", "year", "fuel", "seller_type", "transmission",
            "km_driven", "mileage_mpg", "engine_cc", "max_power_bhp", "seats", "selling_price_mad"]
    df = df[[c for c in keep if c in df.columns]]

    df["age"] = 2025 - df["year"]
    df.drop(columns=["year"], inplace=True)

    return df

def make_pipeline():
    num_features = ["km_driven", "mileage_mpg", "engine_cc", "max_power_bhp", "seats", "age"]
    cat_features = ["company", "model", "fuel", "seller_type", "transmission"]

    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ])

    return preprocessor, num_features, cat_features

def split(df: pd.DataFrame):
    X = df.drop(columns=["selling_price_mad"])
    y = df["selling_price_mad"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
