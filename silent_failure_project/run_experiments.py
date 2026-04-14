"""
run_experiments.py
Full sweep: 2 datasets x 5 methods x 3 failure modes x 10 severity levels
Methods: conformal, ngboost, mcdropout, tabtransformer
Results saved to results/detection_delay_table.csv
Training loss curves saved to results/training_losses.json
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_loader import load_dataset
from src.models import train_xgboost, train_mlp, train_tabtransformer
from src.inject import apply_failure, FAILURE_MODES, SEVERITY_LEVELS
from src.uncertainty import fit_conformal, fit_ngboost, get_uncertainty, METHODS
from src.alarm import run_ks_alarm, accuracy_drop_alpha, detection_delay

os.makedirs("results", exist_ok=True)

DATASETS           = ["pima", "cleveland"]
FAILURE_MODES_LIST = list(FAILURE_MODES.keys())


def run_dataset(dataset_name, all_losses):
    print(f"\n{'='*55}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*55}")

    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler = load_dataset(dataset_name)
    input_dim = X_train.shape[1]

    # ── Train models ──────────────────────────────────────
    print("\n[1/5] Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    print("\n[2/5] Training NGBoost...")
    ngb = fit_ngboost(X_train, y_train)

    print("\n[3/5] Training MC Dropout MLP...")
    mlp, mlp_losses = train_mlp(X_train, y_train, input_dim=input_dim, epochs=150)

    print("\n[4/5] Training TabTransformer (embed=64, nhead=2, attn_drop=0.3, mlp_drop=0.5)...")
    tabt, tabt_losses = train_tabtransformer(
        X_train, y_train, input_dim=input_dim
        # uses tuned defaults: epochs=250, embed_dim=64, nhead=2,
        # dim_feedforward=128, attn_dropout=0.3, mlp_dropout=0.5
    )

    print("\n[5/5] Fitting conformal wrapper...")
    mapie = fit_conformal(xgb, X_train, y_train, X_cal, y_cal)

    # Store loss histories for plotting
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

    # ── Baseline metrics ──────────────────────────────────
    baseline_acc = accuracy_score(y_test, xgb.predict(X_test))
    baseline_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    print(f"\nBaseline  ACC={baseline_acc:.4f}  AUC={baseline_auc:.4f}")

    # Use more passes for DL methods to get stable uncertainty estimates
    N_PASSES = {"conformal": 1, "ngboost": 1, "mcdropout": 100, "tabtransformer": 100}

    # ── Baseline uncertainty distributions ────────────────
    baseline_unc = {m: get_uncertainty(m, models_dict, X_test, n_passes=N_PASSES[m])
                    for m in METHODS}

    # ── Sweep ─────────────────────────────────────────────
    rows = []

    for failure_mode in FAILURE_MODES_LIST:
        print(f"\n  Failure mode: {failure_mode}")

        severity_unc = {m: [] for m in METHODS}
        y_trues, y_preds = [], []

        for alpha in SEVERITY_LEVELS:
            X_corr, y_corr = apply_failure(X_test, y_test, failure_mode, alpha)

            y_pred = xgb.predict(X_corr)
            y_trues.append(y_corr)
            y_preds.append(y_pred)

            for method in METHODS:
                unc = get_uncertainty(method, models_dict, X_corr, n_passes=N_PASSES[method])
                severity_unc[method].append(unc)

        drop_alpha, accs = accuracy_drop_alpha(y_trues, y_preds, baseline_acc)

        for method in METHODS:
            alarm_alpha, ks_stats = run_ks_alarm(
                baseline_unc[method], severity_unc[method]
            )
            delay = detection_delay(alarm_alpha, drop_alpha)

            rows.append({
                "dataset":           dataset_name,
                "failure_mode":      failure_mode,
                "method":            method,
                "alarm_alpha":       alarm_alpha,
                "drop_alpha":        drop_alpha,
                "detection_delay":   delay,
                "baseline_acc":      round(baseline_acc, 4),
            })

            print(f"    {method:15s}  alarm={alarm_alpha}  drop={drop_alpha}  delay={delay}")

    return rows, models_dict, baseline_unc


def main():
    all_rows   = []
    all_losses = {}

    for dataset_name in DATASETS:
        rows, _, _ = run_dataset(dataset_name, all_losses)
        all_rows.extend(rows)

    # Save results CSV
    df = pd.DataFrame(all_rows)
    csv_path = "results/detection_delay_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # Save loss curves for plot_results.py
    losses_path = "results/training_losses.json"
    with open(losses_path, "w") as f:
        json.dump(all_losses, f)
    print(f"Loss curves saved to {losses_path}")


if __name__ == "__main__":
    main()
