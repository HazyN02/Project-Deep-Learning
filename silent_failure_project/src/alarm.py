"""
src/alarm.py
KS-test alarm logic and detection-delay metric.

The alarm fires after ALARM_CONSECUTIVE consecutive KS-test rejections
(p < ALARM_P_THRESHOLD) comparing corrupted vs. clean uncertainty distributions.
All thresholds are imported from config — do NOT redefine them here.
"""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score

from src.config import (
    SEVERITY_LEVELS,
    ALARM_P_THRESHOLD,
    ALARM_CONSECUTIVE,
    ACCURACY_DROP_THRESHOLD,
)


def run_ks_alarm(
    baseline_uncertainty: np.ndarray,
    severity_uncertainties: list[np.ndarray],
    p_threshold: float = ALARM_P_THRESHOLD,
    consecutive: int = ALARM_CONSECUTIVE,
    severity_levels: list[float] | None = None,
) -> tuple[float | None, list[tuple]]:
    """Run KS-test alarm across all severity levels.

    Compares each severity-level uncertainty distribution against the
    clean baseline. Fires after `consecutive` consecutive rejections.

    Args:
        baseline_uncertainty: Uncertainty scores on clean test data, shape (n,).
        severity_uncertainties: List of uncertainty arrays, one per severity level.
            Length must match len(severity_levels).
        p_threshold: KS p-value below which H0 is rejected (default 0.05).
        consecutive: Number of consecutive rejections needed to fire alarm.
        severity_levels: Alpha values matching severity_uncertainties.
            Defaults to config.SEVERITY_LEVELS.

    Returns:
        alarm_alpha: Severity level at which alarm fires, or None if never.
        ks_stats: List of (alpha, ks_stat, p_value) for each severity level.
    """
    if severity_levels is None:
        severity_levels = SEVERITY_LEVELS

    assert len(severity_levels) == len(severity_uncertainties), (
        f"severity_levels length {len(severity_levels)} != "
        f"severity_uncertainties length {len(severity_uncertainties)}"
    )

    ks_stats      = []
    reject_streak = 0
    alarm_alpha   = None

    for alpha, unc in zip(severity_levels, severity_uncertainties):
        stat, p = ks_2samp(baseline_uncertainty, unc)
        ks_stats.append((alpha, float(stat), float(p)))

        if p < p_threshold:
            reject_streak += 1
        else:
            reject_streak = 0

        if reject_streak >= consecutive and alarm_alpha is None:
            alarm_alpha = alpha

    return alarm_alpha, ks_stats


def accuracy_drop_alpha(
    y_true_list: list[np.ndarray],
    y_pred_list: list[np.ndarray],
    baseline_acc: float,
    threshold: float = ACCURACY_DROP_THRESHOLD,
    severity_levels: list[float] | None = None,
) -> tuple[float | None, list[tuple]]:
    """Find the first severity level where accuracy drops meaningfully.

    Args:
        y_true_list: Ground-truth labels per severity level.
        y_pred_list: Model predictions per severity level.
        baseline_acc: Clean-data accuracy to measure drops against.
        threshold: Fraction below baseline that defines a "drop" (default 0.05).
        severity_levels: Alpha values. Defaults to config.SEVERITY_LEVELS.

    Returns:
        drop_alpha: First alpha where accuracy < baseline*(1-threshold), or None.
        accuracies: List of (alpha, accuracy) for each severity level.
    """
    if severity_levels is None:
        severity_levels = SEVERITY_LEVELS

    accuracies = []
    drop_alpha = None

    for alpha, y_true, y_pred in zip(severity_levels, y_true_list, y_pred_list):
        acc = float(accuracy_score(y_true, y_pred))
        accuracies.append((alpha, acc))
        if acc < (baseline_acc - threshold) and drop_alpha is None:
            drop_alpha = alpha

    return drop_alpha, accuracies


def detection_delay(
    alarm_alpha: float | None,
    drop_alpha:  float | None,
) -> float | None:
    """Compute detection delay in severity steps.

    detection_delay = alarm_alpha - drop_alpha

    Interpretation:
        Negative → method warned BEFORE accuracy dropped (early warning, good).
        Positive → method alerted AFTER accuracy dropped (late, bad).
        Zero     → fires exactly at the accuracy-drop point.
        None     → method never alarmed (or accuracy never dropped).

    Args:
        alarm_alpha: Severity level at which the alarm fired, or None.
        drop_alpha:  Severity level at which accuracy dropped, or None.

    Returns:
        Delay as a float rounded to 1 decimal, or None.
    """
    if alarm_alpha is None or drop_alpha is None:
        return None
    return round(float(alarm_alpha) - float(drop_alpha), 1)
