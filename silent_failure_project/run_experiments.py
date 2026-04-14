"""
run_experiments.py
Full sweep: 2 datasets × 3 methods × 3 failure modes × 10 severity levels
Results saved to results/detection_delay_table.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_loader import load_dataset
from src.models import train_xgboost, train_mlp, mc_dropout_predict
from src.inject import apply_failure, FAILURE_MODES, SEVERITY_LEVELS
from src.uncertainty import fit_conformal, fit_ngboost, get_uncertainty, METHODS
from src.alarm import run_ks_alarm, accuracy_drop_alpha, detection_delay

os.makedirs("results", exist_ok=True)

DATASETS      = ["pima", "cleveland"]
FAILURE_MODES_LIST = list(FAILURE_MODES.keys())


def run_dataset(dataset_name):
    print(f"\n{'='*50}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*50}")

    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler = load_dataset(dataset_name)
    input_dim = X_train.shape[1]

    # ── Train models ──────────────────────────────────
    print("\n[1/3] Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    print("\n[2/3] Training NGBoost...")
    ngb = fit_ngboost(X_train, y_train)

    print("\n[3/3] Training MC Dropout MLP...")
    mlp = train_mlp(X_train, y_train, input_dim=input_dim, epochs=150)

    print("\n[4/4] Fitting conformal wrapper...")
    mapie = fit_conformal(xgb, X_train, y_train, X_cal, y_cal)

    models_dict = {"mapie": mapie, "ngboost": ngb, "mlp": mlp}

    # ── Baseline metrics ──────────────────────────────
    baseline_acc = accuracy_score(y_test, xgb.predict(X_test))
    baseline_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"\nBaseline  ACC={baseline_acc:.4f}  AUC={baseline_auc:.4f}")

    # ── Baseline uncertainty distributions ───────────
    baseline_unc = {m: get_uncertainty(m, models_dict, X_test) for m in METHODS}

    # ── Sweep ─────────────────────────────────────────
    rows = []

    for failure_mode in FAILURE_MODES_LIST:
        print(f"\n  Failure mode: {failure_mode}")

        severity_unc   = {m: [] for m in METHODS}
        y_trues, y_preds = [], []

        for alpha in SEVERITY_LEVELS:
            X_corr, y_corr = apply_failure(X_test, y_test, failure_mode, alpha)

            # Predictions for accuracy tracking
            y_pred = xgb.predict(X_corr)
            y_trues.append(y_corr)
            y_preds.append(y_pred)

            # Uncertainty per method
            for method in METHODS:
                unc = get_uncertainty(method, models_dict, X_corr)
                severity_unc[method].append(unc)

        # Accuracy drop
        drop_alpha, accs = accuracy_drop_alpha(y_trues, y_preds, baseline_acc)

        for method in METHODS:
            alarm_alpha, ks_stats = run_ks_alarm(
                baseline_unc[method], severity_unc[method]
            )
            delay = detection_delay(alarm_alpha, drop_alpha)

            rows.append({
                "dataset":      dataset_name,
                "failure_mode": failure_mode,
                "method":       method,
                "alarm_alpha":  alarm_alpha,
                "drop_alpha":   drop_alpha,
                "detection_delay": delay,
                "baseline_acc": round(baseline_acc, 4),
            })

            print(f"    {method:12s}  alarm={alarm_alpha}  drop={drop_alpha}  delay={delay}")

    return rows, models_dict, baseline_unc


def main():
    all_rows = []
    for dataset_name in DATASETS:
        rows, _, _ = run_dataset(dataset_name)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = "results/detection_delay_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
