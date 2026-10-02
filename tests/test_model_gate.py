"""MLOps quality gate: F1 threshold, inference latency, output schema."""
import statistics
import time

import mlflow.sklearn
import pytest

from src.data import get_splits
from src.train import CHAMPION_ALIAS, MODEL_NAME, run_experiments

F1_THRESHOLD = 0.88
LATENCY_LIMIT_MS = 30.0


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Train in an isolated MLflow store so the gate runs on a fresh CI box."""
    root = tmp_path_factory.mktemp("mlflow")
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    results = run_experiments(
        tracking_uri=uri, artifact_location=(root / "artifacts").as_uri()
    )
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")
    return results, model


def test_metric_threshold_gate(pipeline):
    results, _ = pipeline
    best_f1 = results["val_f1_macro"].max()
    assert best_f1 >= F1_THRESHOLD, f"Validation macro F1 {best_f1:.4f} < {F1_THRESHOLD}"


def test_inference_latency_gate(pipeline):
    _, model = pipeline
    _, X_test, _, _ = get_splits()
    model.predict(X_test)  # warm-up
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        model.predict(X_test)
        timings.append((time.perf_counter() - start) * 1000)
    median_ms = statistics.median(timings)
    assert median_ms <= LATENCY_LIMIT_MS, f"Latency {median_ms:.2f} ms > {LATENCY_LIMIT_MS} ms"


def test_output_schema_integrity(pipeline):
    _, model = pipeline
    _, X_test, _, _ = get_splits()
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds.tolist()) <= {0, 1, 2}
