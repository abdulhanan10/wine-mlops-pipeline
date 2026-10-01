"""Data loading, validation and splitting for the Wine dataset."""
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
EXPECTED_FEATURES = 13


def load_dataset():
    """Return the Wine features (DataFrame) and target (Series)."""
    bunch = load_wine(as_frame=True)
    return bunch.data, bunch.target


def validate_data(X: pd.DataFrame, y: pd.Series) -> bool:
    """Raise ValueError if the data has nulls or the wrong feature count."""
    if X.isnull().values.any() or y.isnull().any():
        raise ValueError("Data contains null values.")
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, got {X.shape[1]}."
        )
    return True


def get_splits(test_size: float = 0.2, seed: int = SEED):
    """Validate the data and return a stratified train/test split."""
    X, y = load_dataset()
    validate_data(X, y)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
