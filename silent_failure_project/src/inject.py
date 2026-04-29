"""
src/inject.py
Synthetic failure-injection functions for the silent-failure benchmark.

Each function accepts severity α ∈ [0.0, 1.0]:
  α = 0.0  →  no change (clean data)
  α = 1.0  →  maximum distortion

All functions are pure (no side effects) and accept random_state for reproducibility.
SEVERITY_LEVELS is imported from config — do NOT redefine it here.
"""

import numpy as np
from src.config import SEVERITY_LEVELS, RANDOM_SEED   # single source of truth


# ── Covariate Shift ────────────────────────────────────────────────────────

def covariate_shift(
    X: np.ndarray,
    alpha: float,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Resample rows using exponential importance weights on feature 0.

    Higher alpha increases the skew toward high-feature-0 samples,
    simulating gradual population drift (e.g., older / higher-BMI patients).

    Args:
        X: Feature array, shape (n, d).
        alpha: Severity in [0, 1]. Alpha=0 → identity; alpha=1 → maximum skew.
        random_state: Seed for reproducibility.

    Returns:
        X_shifted: Resampled feature array, same shape as X.
    """
    rng = np.random.default_rng(random_state)
    weights = np.exp(alpha * X[:, 0])
    weights /= weights.sum()
    idx = rng.choice(len(X), size=len(X), replace=True, p=weights)
    return X[idx]


def inject_covariate_shift(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply covariate_shift and keep y aligned with the resampled rows.

    Args:
        X: Feature array, shape (n, d).
        y: Label array, shape (n,).
        alpha: Severity in [0, 1].
        random_state: Seed for reproducibility.

    Returns:
        (X_shifted, y_shifted): Resampled feature and label arrays.
    """
    rng = np.random.default_rng(random_state)
    weights = np.exp(alpha * X[:, 0])
    weights /= weights.sum()
    idx = rng.choice(len(X), size=len(X), replace=True, p=weights)
    return X[idx], y[idx]


# ── Label Noise ────────────────────────────────────────────────────────────

def label_noise(
    y: np.ndarray,
    alpha: float,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Randomly flip a fraction alpha of labels.

    NOTE: This failure mode is undetectable by feature-distribution-based
    uncertainty monitors because X is unchanged.

    Args:
        y: Binary label array, shape (n,).
        alpha: Fraction of labels to flip, in [0, 1].
        random_state: Seed for reproducibility.

    Returns:
        y_noisy: Label array with alpha fraction flipped.
    """
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(alpha * len(y))
    if n_flip > 0:
        flip_idx = rng.choice(len(y), size=n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
    return y_noisy


def inject_label_noise(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply label_noise; X is returned unchanged.

    Args:
        X: Feature array, shape (n, d) — returned as-is.
        y: Binary label array, shape (n,).
        alpha: Fraction of labels to flip, in [0, 1].
        random_state: Seed for reproducibility.

    Returns:
        (X, y_noisy): X unchanged, labels partially flipped.
    """
    return X, label_noise(y, alpha, random_state=random_state)


# ── Feature Missingness ────────────────────────────────────────────────────

def feature_masking(
    X: np.ndarray,
    alpha: float,
    col: int = 0,
    strategy: str = "mean",
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """Mask feature column `col` for a fraction alpha of rows.

    NOTE: With strategy='mean', masking a StandardScaler-normalized column
    to 0.0 is equivalent to imputing the training mean — no distributional
    shift occurs and accuracy is unaffected. Use strategy='sentinel' for
    an out-of-range value that forces a real shift.

    Args:
        X: Feature array, shape (n, d).
        alpha: Fraction of rows to mask, in [0, 1].
        col: Index of column to mask (default 0).
        strategy: 'mean' (impute 0.0 after scaling), or 'sentinel'
                  (impute -3.0, outside training distribution).
        random_state: Seed for reproducibility.

    Returns:
        X_masked: Feature array with column `col` masked for alpha rows.
    """
    rng = np.random.default_rng(random_state)
    X_out = X.copy()
    n_mask = int(alpha * len(X))
    if n_mask > 0:
        mask_idx = rng.choice(len(X), size=n_mask, replace=False)
        fill_val = 0.0 if strategy == "mean" else -3.0
        X_out[mask_idx, col] = fill_val
    return X_out


def inject_feature_missingness(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    col: int = 0,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply feature_masking; y is returned unchanged.

    Args:
        X: Feature array, shape (n, d).
        y: Label array, shape (n,) — returned as-is.
        alpha: Fraction of rows to mask, in [0, 1].
        col: Column to mask (default 0).
        random_state: Seed for reproducibility.

    Returns:
        (X_masked, y): Masked features, labels unchanged.
    """
    return feature_masking(X, alpha, col=col, random_state=random_state), y


# ── Unified dispatch ───────────────────────────────────────────────────────

FAILURE_MODES = {
    "covariate_shift":     inject_covariate_shift,
    "label_noise":         inject_label_noise,
    "feature_missingness": inject_feature_missingness,
}


def apply_failure(
    X: np.ndarray,
    y: np.ndarray,
    mode: str,
    alpha: float,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a named failure mode at a given severity level.

    Args:
        X: Feature array, shape (n, d).
        y: Label array, shape (n,).
        mode: One of 'covariate_shift', 'label_noise', 'feature_missingness'.
        alpha: Severity in [0, 1].
        random_state: Seed for reproducibility.

    Returns:
        (X_corrupted, y_corrupted): Arrays after failure injection.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode not in FAILURE_MODES:
        raise ValueError(
            f"Unknown failure mode '{mode}'. "
            f"Choose from: {list(FAILURE_MODES.keys())}"
        )
    return FAILURE_MODES[mode](X, y, alpha, random_state=random_state)
