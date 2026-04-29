"""
run_experiments.py
Full sweep: 2 datasets x 4 methods x 3 failure modes x 10 severity levels.

Outputs:
  results/detection_delay_table.csv     -- full 24-row sweep results
  results/detection_delay_summary.csv   -- pivot by failure_mode x method
  results/metrics_comparison.csv        -- ECE + Brier per method per dataset
  results/training_losses.json          -- DL training curves for plot_results.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Seeds first, before any import that touches random state ───────────────
from src.seed_everything import seed_everything
from src.config import (
    RANDOM_SEED, RESULTS_PATH, DATASETS, METHODS, METHOD_LABELS,
    SEVERITY_LEVELS, N_MC_PASSES,
)
seed_everything(RANDOM_SEED)

from src.data_loader import load_dataset
from src.models import train_xgboost, train_mlp, train_tabtransformer
from src.inject import apply_failure, FAILURE_MODES
from src.uncertainty import fit_conformal, fit_ngboost, get_uncertainty
from src.alarm import run_ks_alarm, accuracy_drop_alpha, detection_delay
from src.evaluation import (
    expected_calibration_error, brier_score, save_results,
)

os.makedirs(RESULTS_PATH, exist_ok=True)

FAILURE_MODES_LIST = list(FAILURE_MODES.keys())


def run_dataset(dataset_name, all_losses):
    """Train all models and run the full severity sweep for one dataset.

    Args:
        dataset_name: 'pima' or 'cleveland'.
        all_losses: Dict to accumulate DL training loss histories (mutated).

    Returns:
        rows: List of result dicts (one per method x failure_mode).
        models_dict: Fitted model objects.
        baseline_unc: Dict method -> uncertainty array on clean test data.
        xgb: Fitted XGBClassifier.
        X_test, y_test: Clean test split.
    """
    print(f"\n{'='*55}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*55}")

    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler = load_dataset(dataset_name)
    input_dim = X_train.shape[1]

    # ── Train models ──────────────────────────────────────────────────────
    print("\n[1/5] Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    print("\n[2/5] Training NGBoost...")
    ngb = fit_ngboost(X_train, y_train)

    print("\n[3/5] Training MC Dropout MLP...")
    mlp, mlp_losses = train_mlp(X_train, y_train, input_dim=input_dim)

    print("\n[4/5] Training TabTransformer (embed=64, nhead=2, attn_drop=0.3, mlp_drop=0.5)...")
    tabt, tabt_losses = train_tabtransformer(X_train, y_train, input_dim=input_dim)

    print("\n[5/5] Fitting conformal wrapper...")
    mapie = fit_conformal(xgb, X_train, y_train, X_cal, y_cal)

    all_losses[dataset_name] = {
        "mlp":            mlp_losses,
        "tabtransformer": tabt_losses,
    }

    models_dict = {
        "mapie":          mapie,
        "ngboost":        ngb,
        "mlp":            mlp,
        "tabtransformer": tabt,
    }

    # ── Baseline metrics ──────────────────────────────────────────────────
    baseline_acc = float(accuracy_score(y_test, xgb.predict(X_test)))
    baseline_auc = float(roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]))
    print(f"\n  Baseline  ACC={baseline_acc:.4f}  AUC={baseline_auc:.4f}")

    # ── Baseline uncertainty distributions ───────────────────────────────
    baseline_unc = {
        m: get_uncertainty(m, models_dict, X_test, n_passes=N_MC_PASSES[m])
        for m in METHODS
    }

    # ── Sweep ─────────────────────────────────────────────────────────────
    rows = []
    for failure_mode in FAILURE_MODES_LIST:
        print(f"\n  Failure mode: {failure_mode}")

        severity_unc = {m: [] for m in METHODS}
        y_trues, y_preds = [], []

        for alpha in SEVERITY_LEVELS:
            X_corr, y_corr = apply_failure(X_test, y_test, failure_mode, alpha)
            y_trues.append(y_corr)
            y_preds.append(xgb.predict(X_corr))

            for method in METHODS:
                unc = get_uncertainty(
                    method, models_dict, X_corr,
                    n_passes=N_MC_PASSES[method],
                )
                severity_unc[method].append(unc)

        drop_alpha, _ = accuracy_drop_alpha(y_trues, y_preds, baseline_acc)

        for method in METHODS:
            alarm_alpha, _ = run_ks_alarm(baseline_unc[method], severity_unc[method])
            delay = detection_delay(alarm_alpha, drop_alpha)

            rows.append({
                "dataset":         dataset_name,
                "failure_mode":    failure_mode,
                "method":          method,
                "alarm_alpha":     alarm_alpha,
                "drop_alpha":      drop_alpha,
                "detection_delay": delay,
                "baseline_acc":    round(baseline_acc, 4),
                "baseline_auc":    round(baseline_auc, 4),
            })
            print(f"    {method:15s}  alarm={alarm_alpha}  drop={drop_alpha}  delay={delay}")

    return rows, models_dict, baseline_unc, xgb, X_test, y_test


def main():
    all_rows     = []
    all_losses   = {}
    metrics_rows = []

    for dataset_name in DATASETS:
        rows, models_dict, baseline_unc, xgb, X_test, y_test = run_dataset(
            dataset_name, all_losses
        )
        all_rows.extend(rows)

        # ECE + Brier for this dataset
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        ece   = expected_calibration_error(xgb_probs, y_test)
        brier = brier_score(xgb_probs, y_test)

        baseline_acc = rows[0]["baseline_acc"] if rows else None
        baseline_auc = rows[0]["baseline_auc"] if rows else None

        for method in METHODS:
            unc = baseline_unc.get(method, np.array([]))
            metrics_rows.append({
                "dataset":          dataset_name,
                "method":           method,
                "method_label":     METHOD_LABELS[method],
                "baseline_acc":     baseline_acc,
                "baseline_auc":     baseline_auc,
                "mean_uncertainty": round(float(unc.mean()), 6) if len(unc) else None,
                "ECE":              round(ece, 4),
                "Brier_Score":      round(brier, 4),
            })

    # ── Save full sweep CSV ────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    save_results(df, "detection_delay_table.csv")

    # ── Save detection delay summary pivot ────────────────────────────────
    pivot = df.pivot_table(
        index=["dataset", "failure_mode"],
        columns="method",
        values="detection_delay",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    save_results(pivot, "detection_delay_summary.csv")

    # ── Save metrics comparison ────────────────────────────────────────────
    save_results(pd.DataFrame(metrics_rows), "metrics_comparison.csv")

    # ── Save DL loss curves ────────────────────────────────────────────────
    losses_path = os.path.join(RESULTS_PATH, "training_losses.json")
    with open(losses_path, "w") as f:
        json.dump(all_losses, f)
    print(f"\n  Saved: {losses_path}")

    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
