"""
src/evaluation.py
Unified evaluation utilities for the Silent Failure Detection benchmark.

Provides:
  - expected_calibration_error(probs, y_true)
  - brier_score(probs, y_true)
  - evaluate_all_methods(methods_info, X_test, y_test)
  - build_summary_table(rows, methods)  -- formats the detection delay table
  - save_results(df, path)
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import RESULTS_PATH, METHODS, METHOD_LABELS


# ── Calibration Metrics ────────────────────────────────────────────────────

def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures how well predicted probabilities match actual outcome rates.
    A perfectly calibrated model has ECE = 0.

    Args:
        probs: Predicted probabilities for the positive class, shape (n,).
        y_true: Binary ground-truth labels, shape (n,).
        n_bins: Number of equal-width probability bins (default 10).

    Returns:
        ECE as a float in [0, 1].
    """
    probs  = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc  = y_true[mask].mean()
            bin_conf = probs[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Brier Score: mean squared error between predicted probabilities and labels.

    Range [0, 1]. Lower is better. A perfectly calibrated model achieves
    the Brier score equal to 1 - prevalence at baseline.

    Args:
        probs: Predicted probabilities for the positive class, shape (n,).
        y_true: Binary ground-truth labels, shape (n,).

    Returns:
        Brier score as a float in [0, 1].
    """
    probs  = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((probs - y_true) ** 2))


# ── Unified Method Evaluation ─────────────────────────────────────────────

def evaluate_all_methods(
    models_dict: dict,
    xgb_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_passes_map: dict | None = None,
) -> pd.DataFrame:
    """Run all four uncertainty methods and return a unified comparison DataFrame.

    Computes accuracy, mean uncertainty, ECE, and Brier score for each method.
    Note: accuracy and Brier/ECE are computed from the XGBoost production model
    (the same model whose predictions are monitored in deployment). Uncertainty
    scores come from each respective method.

    Args:
        models_dict: Dict with keys 'mapie', 'ngboost', 'mlp', 'tabtransformer'.
        xgb_model: Fitted XGBClassifier used for accuracy/calibration metrics.
        X_test: Test features, shape (n, d).
        y_test: Test labels, shape (n,).
        n_passes_map: Optional dict method -> int for MC passes.
            Defaults to config.N_MC_PASSES.

    Returns:
        DataFrame with columns:
            Method, Accuracy, Mean_Uncertainty, ECE, Brier_Score
    """
    from src.config import N_MC_PASSES
    from src.uncertainty import get_uncertainty

    if n_passes_map is None:
        n_passes_map = N_MC_PASSES

    # XGBoost production model metrics (same model being monitored)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    accuracy  = float(accuracy_score(y_test, xgb_preds))
    ece       = expected_calibration_error(xgb_probs, y_test)
    brier     = brier_score(xgb_probs, y_test)

    rows = []
    for method in METHODS:
        unc = get_uncertainty(
            method, models_dict, X_test,
            n_passes=n_passes_map.get(method, 50),
        )
        assert not np.isnan(unc).any(), (
            f"NaN uncertainty scores from method '{method}' on clean data. "
            "Check model training."
        )
        rows.append({
            "Method":           METHOD_LABELS.get(method, method),
            "Accuracy":         round(accuracy, 4),
            "Mean_Uncertainty": round(float(unc.mean()), 6),
            "ECE":              round(ece, 4),
            "Brier_Score":      round(brier, 4),
        })

    return pd.DataFrame(rows)


# ── Summary Table Builder ──────────────────────────────────────────────────

def build_summary_table(sweep_rows: list[dict]) -> pd.DataFrame:
    """Build the final detection delay summary table from sweep results.

    Aggregates per-method results across failure modes and formats for saving.

    Args:
        sweep_rows: List of dicts from the experiment sweep, each with keys:
            dataset, failure_mode, method, alarm_alpha, drop_alpha,
            detection_delay, baseline_acc.

    Returns:
        DataFrame sorted by dataset, failure_mode, method.
    """
    df = pd.DataFrame(sweep_rows)
    df = df.sort_values(["dataset", "failure_mode", "method"]).reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, filename: str) -> str:
    """Save a DataFrame to results/ and print confirmation.

    Args:
        df: DataFrame to save.
        filename: Filename within the results directory (e.g. 'summary.csv').

    Returns:
        Absolute path of saved file.
    """
    os.makedirs(RESULTS_PATH, exist_ok=True)
    path = os.path.join(RESULTS_PATH, filename)
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")
    return path
