import argparse
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .preprocessing import load_raw, preprocess, split
from .pipeline import build_model, save_model

def train(args):
    df = load_raw()
    df = preprocess(df)
    X_train, X_test, y_train, y_test = split(df)

    pipeline = build_model(n_estimators=args.n_estimators)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    # Compatibilité toutes versions sklearn
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.2f} MAD | MAE: {mae:.2f} MAD | R2: {r2:.3f}")

    save_model(pipeline)
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    args = parser.parse_args()
    train(args)
