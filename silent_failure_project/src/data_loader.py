import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def load_pima():
    """Pima Indians Diabetes dataset.

    UCI id=34 is restricted from programmatic import; falls back to the
    identical dataset via sklearn/OpenML (name='diabetes', version=1).
    """
    try:
        dataset = fetch_ucirepo(id=34)
        X = dataset.data.features.copy()
        y = dataset.data.targets.copy().values.ravel()
    except Exception:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name='diabetes', version=1, as_frame=True)
        X = data.data.copy()
        y = data.target.values
    # Binarize: tested_positive=1, tested_negative=0
    y = (y == 'tested_positive').astype(int)
    return X, y

def load_cleveland():
    """Cleveland Heart Disease dataset (UCI ID: 45)"""
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy().values.ravel()
    # Binarize: >0 means disease present
    y = (y > 0).astype(int)
    # Drop rows with missing values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    return X, y

def get_splits(X, y, test_size=0.3, cal_size=0.15, random_state=42):
    """
    Returns train, calibration, and test splits.
    Calibration split is used for conformal prediction.
    """
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + cal_size), random_state=random_state, stratify=y
    )
    cal_frac = cal_size / (test_size + cal_size)
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - cal_frac), random_state=random_state, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_cal   = scaler.transform(X_cal)
    X_test  = scaler.transform(X_test)

    return X_train, X_cal, X_test, y_train, y_cal, y_test, scaler

def load_dataset(name="pima"):
    if name == "pima":
        X, y = load_pima()
    elif name == "cleveland":
        X, y = load_cleveland()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return get_splits(X, y)
