"""Train RF and GBM candidates, track them in MLflow, register the champion."""
import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, get_splits

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
N_FOLDS = 5

# Each family: (estimator class, list of >= 3 hyperparameter configurations)
SEARCH_SPACE = {
    "RandomForest": (
        RandomForestClassifier,
        [
            {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2},
            {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
            {"n_estimators": 100, "max_depth": None, "min_samples_split": 4},
            {"n_estimators": 200, "max_depth": 8, "min_samples_split": 2},
        ],
    ),
    "GradientBoosting": (
        GradientBoostingClassifier,
        [
            {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2},
            {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
            {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
            {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 2},
        ],
    ),
}

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "neg_log_loss": "neg_log_loss",
}


def run_experiments(tracking_uri=DEFAULT_TRACKING_URI, artifact_location=None):
    """Run every configuration, register the best one, return a results table."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_location)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = get_splits()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    rows = []
    for family, (estimator_cls, configs) in SEARCH_SPACE.items():
        for i, params in enumerate(configs, start=1):
            model = estimator_cls(random_state=SEED, **params)
            with mlflow.start_run(run_name=f"{family}-cfg{i}") as run:
                mlflow.set_tags({
                    "model_family": family,
                    "stage": "hyperparameter-tuning",
                    "dataset": "sklearn.load_wine",
                })
                mlflow.log_params({**params, "cv_folds": N_FOLDS, "seed": SEED})

                scores = cross_validate(
                    model, X_train, y_train, cv=cv, scoring=SCORING,
                    return_train_score=True,
                )
                metrics = {
                    "train_f1_macro": scores["train_f1_macro"].mean(),
                    "val_f1_macro": scores["test_f1_macro"].mean(),
                    "train_accuracy": scores["train_accuracy"].mean(),
                    "val_accuracy": scores["test_accuracy"].mean(),
                    "train_log_loss": -scores["train_neg_log_loss"].mean(),
                    "val_log_loss": -scores["test_neg_log_loss"].mean(),
                }
                mlflow.log_metrics(metrics)

                model.fit(X_train, y_train)
                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_train.head(5),
                )
                rows.append({
                    "run_id": run.info.run_id,
                    "run_name": f"{family}-cfg{i}",
                    "family": family,
                    "params": str(params),
                    **metrics,
                })

    results = pd.DataFrame(rows)
    best = results.loc[results["val_f1_macro"].idxmax()]

    version = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    client.set_tag(best["run_id"], "champion", "true")
    print(f"Champion: {best['run_name']} (val macro F1 = {best['val_f1_macro']:.4f}) "
          f"-> {MODEL_NAME} v{version.version} @{CHAMPION_ALIAS}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train and track Wine classifiers.")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    results = run_experiments(tracking_uri=args.tracking_uri)
    os.makedirs("results", exist_ok=True)
    results.drop(columns=["run_id"]).round(4).to_csv("results/search_results.csv", index=False)
    print(results.drop(columns=["run_id", "params"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
