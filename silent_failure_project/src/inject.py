import numpy as np
from sklearn.preprocessing import StandardScaler

SEVERITY_LEVELS = [round(i * 0.1, 1) for i in range(10)]  # 0.0 to 0.9


def inject_covariate_shift(X, y, alpha, random_state=42):
    """
    Resamples test set to skew the first feature (proxy for age/BMI)
    using exponential importance weights. Higher alpha = more skew.
    """
    rng = np.random.default_rng(random_state)
    feature = X[:, 0]
    weights = np.exp(alpha * feature)
    weights /= weights.sum()
    n = len(X)
    idx = rng.choice(n, size=n, replace=True, p=weights)
    return X[idx], y[idx]


def inject_label_noise(X, y, alpha, random_state=42):
    """
    Randomly flips a fraction alpha of labels.
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(alpha * len(y))
    flip_idx = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return X, y_noisy


def inject_feature_missingness(X, y, alpha, col=0, random_state=42):
    """
    Progressively masks column `col` for alpha fraction of rows,
    imputing with zero (mean after standardization).
    """
    rng = np.random.default_rng(random_state)
    X_missing = X.copy()
    n_mask = int(alpha * len(X))
    mask_idx = rng.choice(len(X), size=n_mask, replace=False)
    X_missing[mask_idx, col] = 0.0  # 0 = training mean after StandardScaler
    return X_missing, y


FAILURE_MODES = {
    "covariate_shift": inject_covariate_shift,
    "label_noise": inject_label_noise,
    "feature_missingness": inject_feature_missingness,
}


def apply_failure(X, y, mode, alpha, random_state=42):
    """
    Apply a named failure mode at a given severity level alpha.
    Returns (X_corrupted, y_corrupted).
    """
    if mode not in FAILURE_MODES:
        raise ValueError(f"Unknown failure mode: {mode}. Choose from {list(FAILURE_MODES.keys())}")
    return FAILURE_MODES[mode](X, y, alpha, random_state=random_state)
