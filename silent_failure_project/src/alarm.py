import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score


SEVERITY_LEVELS = [round(i * 0.1, 1) for i in range(10)]  # 0.0 to 0.9
ALARM_P_THRESHOLD = 0.05
ALARM_CONSECUTIVE  = 3  # consecutive rejections needed


def run_ks_alarm(
    baseline_uncertainty,
    severity_uncertainties,
    p_threshold=ALARM_P_THRESHOLD,
    consecutive=ALARM_CONSECUTIVE,
):
    """
    For each severity level, run a KS test comparing the uncertainty
    distribution against the clean baseline.

    Returns:
        alarm_alpha: severity level at which alarm fires (None if never)
        ks_stats: list of (stat, p_value) per severity level
    """
    ks_stats = []
    reject_streak = 0
    alarm_alpha = None

    for i, (alpha, unc) in enumerate(zip(SEVERITY_LEVELS, severity_uncertainties)):
        stat, p = ks_2samp(baseline_uncertainty, unc)
        ks_stats.append((alpha, stat, p))

        if p < p_threshold:
            reject_streak += 1
        else:
            reject_streak = 0

        if reject_streak >= consecutive and alarm_alpha is None:
            alarm_alpha = alpha

    return alarm_alpha, ks_stats


def accuracy_drop_alpha(y_true_list, y_pred_list, baseline_acc, threshold=0.05):
    """
    Finds the first severity level where accuracy drops more than `threshold`
    below the clean baseline.

    Returns:
        drop_alpha: severity level of first visible drop (None if never)
        accuracies: list of (alpha, accuracy)
    """
    accuracies = []
    drop_alpha = None

    for alpha, y_true, y_pred in zip(SEVERITY_LEVELS, y_true_list, y_pred_list):
        acc = accuracy_score(y_true, y_pred)
        accuracies.append((alpha, acc))
        if acc < (baseline_acc - threshold) and drop_alpha is None:
            drop_alpha = alpha

    return drop_alpha, accuracies


def detection_delay(alarm_alpha, drop_alpha):
    """
    detection_delay = alarm_alpha - drop_alpha
    Negative = early warning (good).
    Positive = method alerted after accuracy already dropped (bad).
    None     = method never alarmed.
    """
    if alarm_alpha is None:
        return None
    if drop_alpha is None:
        # Alarm fired but accuracy never dropped — could be false positive
        return None
    return round(alarm_alpha - drop_alpha, 1)
