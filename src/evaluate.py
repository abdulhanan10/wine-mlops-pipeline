"""Load the registered champion model and score it on the held-out test split."""
import argparse

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_splits
from src.train import CHAMPION_ALIAS, DEFAULT_TRACKING_URI, MODEL_NAME


def evaluate(tracking_uri=DEFAULT_TRACKING_URI):
    mlflow.set_tracking_uri(tracking_uri)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    _, X_test, _, y_test = get_splits()

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    return {
        "test_f1_macro": f1_score(y_test, preds, average="macro"),
        "test_accuracy": accuracy_score(y_test, preds),
        "test_log_loss": log_loss(y_test, proba, labels=[0, 1, 2]),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the champion model.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()
    for name, value in evaluate(args.tracking_uri).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
