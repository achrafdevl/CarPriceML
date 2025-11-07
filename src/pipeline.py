from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load
from .preprocessing import make_pipeline
from pathlib import Path
from .config import MODEL_DIR

MODEL_FILE = MODEL_DIR / "rf_model.joblib"

def build_model(n_estimators=100, random_state=42):
    preprocessor, _, _ = make_pipeline()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline

def save_model(pipeline, path: Path = MODEL_FILE):
    dump(pipeline, path)

def load_model(path: Path = MODEL_FILE):
    return load(path)
