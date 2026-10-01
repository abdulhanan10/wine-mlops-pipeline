import pytest

from src.data import get_splits, load_dataset, validate_data


def test_dataset_shape():
    X, y = load_dataset()
    assert X.shape == (178, 13)
    assert set(y.unique()) == {0, 1, 2}


def test_validation_passes_on_clean_data():
    X, y = load_dataset()
    assert validate_data(X, y) is True


def test_validation_rejects_nulls():
    X, y = load_dataset()
    X = X.copy()
    X.iloc[0, 0] = None
    with pytest.raises(ValueError):
        validate_data(X, y)


def test_validation_rejects_wrong_feature_count():
    X, y = load_dataset()
    with pytest.raises(ValueError):
        validate_data(X.iloc[:, :12], y)


def test_split_is_80_20_and_stratified():
    X_train, X_test, y_train, y_test = get_splits()
    assert len(X_train) == 142 and len(X_test) == 36
    full = load_dataset()[1].value_counts(normalize=True)
    for cls in (0, 1, 2):
        assert abs(y_train.value_counts(normalize=True)[cls] - full[cls]) < 0.02


def test_split_is_reproducible():
    a = get_splits()[1]
    b = get_splits()[1]
    assert a.index.equals(b.index)
