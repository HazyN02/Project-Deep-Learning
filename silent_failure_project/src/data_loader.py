"""
src/data_loader.py
Dataset loading and train/calibration/test splitting.

Supports:
  - Pima Indians Diabetes (UCI id=34, OpenML fallback)
  - Cleveland Heart Disease (UCI id=45)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED


# ── Pima Indians Diabetes ──────────────────────────────────────────────────

def load_pima() -> tuple[np.ndarray, np.ndarray]:
    """Load the Pima Indians Diabetes dataset.

    Tries UCI mirror first; falls back to OpenML on network/access errors.
    Labels are binarized: tested_positive → 1, tested_negative → 0.

    Returns:
        X: Feature array, shape (768, 8), float32.
        y: Binary label array, shape (768,), int32.

    Raises:
        RuntimeError: If both UCI and OpenML sources fail.
    """
    X, y_raw = None, None

    # Primary source: UCI
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=34)
        X = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy().values.ravel()
    except Exception:
        pass

    # Fallback: OpenML
    if X is None:
        try:
            from sklearn.datasets import fetch_openml
            data = fetch_openml(name="diabetes", version=1, as_frame=True)
            X = data.data.copy()
            y_raw = data.target.values
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Pima dataset from both UCI (id=34) and "
                f"OpenML (name='diabetes'). Original error: {exc}"
            ) from exc

    # Binarize labels — handle both string and numeric encodings
    if hasattr(y_raw[0], "lower"):
        y = (np.array([str(v).lower() for v in y_raw]) == "tested_positive").astype(np.int32)
    else:
        y = (np.array(y_raw) > 0).astype(np.int32)

    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    X = X.apply(pd.to_numeric, errors="coerce")

    # Assertions
    assert X.shape == (768, 8), (
        f"Pima: expected shape (768, 8), got {X.shape}. "
        "Dataset version may have changed."
    )
    assert set(np.unique(y)) <= {0, 1}, f"Pima: unexpected label values {np.unique(y)}"
    pos_rate = y.mean()
    assert 0.3 <= pos_rate <= 0.4, (
        f"Pima: positive-class rate {pos_rate:.3f} outside expected range [0.30, 0.40]. "
        "Check label binarization."
    )

    nan_count = int(X.isnull().sum().sum())
    if nan_count > 0:
        print(f"[data_loader] Pima: {nan_count} NaN values detected — imputing with column means.")
        X = X.fillna(X.mean())

    return X.values.astype(np.float32), y


# ── Cleveland Heart Disease ────────────────────────────────────────────────

def load_cleveland() -> tuple[np.ndarray, np.ndarray]:
    """Load the Cleveland Heart Disease dataset (UCI id=45).

    Labels are binarized: num > 0 → 1 (disease present), 0 → 0.
    Rows with NaN values are dropped.

    Returns:
        X: Feature array, shape (≈297, 13), float32.
        y: Binary label array, int32.

    Raises:
        RuntimeError: If UCI fetch fails.
    """
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy().values.ravel()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Cleveland dataset (UCI id=45): {exc}"
        ) from exc

    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    X = X.apply(pd.to_numeric, errors="coerce")

    y = (np.array(y_raw, dtype=float) > 0).astype(np.int32)

    # Drop rows with NaN
    mask = ~X.isnull().any(axis=1).values
    X = X[mask]
    y = y[mask]

    assert X.shape[1] == 13, (
        f"Cleveland: expected 13 features, got {X.shape[1]}."
    )
    assert set(np.unique(y)) <= {0, 1}, f"Cleveland: unexpected label values {np.unique(y)}"
    assert len(X) >= 200, f"Cleveland: only {len(X)} rows after NaN drop — check source."

    return X.values.astype(np.float32), y


# ── Splitting ──────────────────────────────────────────────────────────────

def get_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.30,
    cal_size: float  = 0.15,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """Split data into stratified train / calibration / test sets and scale.

    The calibration split is reserved for MAPIE conformal calibration.
    Fitting StandardScaler on train only, then transforming cal + test.

    Args:
        X: Feature array, shape (n, d), float32.
        y: Label array, shape (n,), int32.
        test_size: Fraction of total data for the test set.
        cal_size: Fraction of total data for the calibration set.
        random_state: Random seed for splits.

    Returns:
        X_train, X_cal, X_test, y_train, y_cal, y_test, scaler
        All arrays are float32 / int32. Scaler is fitted on X_train.
    """
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    assert not np.isnan(X).any(), "X contains NaN values before splitting."
    assert not np.isinf(X).any(), "X contains Inf values before splitting."
    assert len(X) == len(y), "X and y length mismatch."

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + cal_size),
        random_state=random_state,
        stratify=y,
    )
    cal_frac = cal_size / (test_size + cal_size)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1.0 - cal_frac),
        random_state=random_state,
        stratify=y_temp,
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal   = scaler.transform(X_cal)
    X_test  = scaler.transform(X_test)

    return X_train, X_cal, X_test, y_train, y_cal, y_test, scaler


def load_dataset(name: str = "pima") -> tuple:
    """Load and split a named dataset.

    Args:
        name: One of 'pima' or 'cleveland'.

    Returns:
        X_train, X_cal, X_test, y_train, y_cal, y_test, scaler

    Raises:
        ValueError: If name is not recognized.
    """
    if name == "pima":
        X, y = load_pima()
    elif name == "cleveland":
        X, y = load_cleveland()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: pima, cleveland")
    return get_splits(X, y)
