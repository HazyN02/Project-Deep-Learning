"""
plot_results.py
Generates all figures from results/ (written by run_experiments.py).

Outputs:
  results/heatmap_{dataset}.png
  results/alarm_vs_drop_{dataset}.png
  results/training_curves.png          (combined MLP + TabTransformer)
  results/training_curves_{dataset}.png (per-dataset)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    RESULTS_PATH, METHOD_LABELS, FAILURE_LABELS, DATASETS,
)

os.makedirs(RESULTS_PATH, exist_ok=True)


# ── Detection Delay Heatmap ────────────────────────────────────────────────

def plot_detection_delay_heatmap(df: pd.DataFrame, dataset_name: str) -> None:
    """Save a heatmap of detection delay (failure_mode x method).

    Args:
        df: Full detection_delay_table DataFrame.
        dataset_name: 'pima' or 'cleveland'.
    """
    subset = df[df["dataset"] == dataset_name].copy()
    subset["method_label"]  = subset["method"].map(METHOD_LABELS)
    subset["failure_label"] = subset["failure_mode"].map(FAILURE_LABELS)

    pivot = subset.pivot(
        index="failure_label",
        columns="method_label",
        values="detection_delay",
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Detection Delay (a_alarm - a_drop)\nNegative = early warning"},
    )
    ax.set_title(
        f"Detection Delay Heatmap -- {dataset_name.capitalize()} Dataset\n"
        "Negative = method warned before accuracy drop",
        fontsize=11,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, f"heatmap_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Alarm vs Drop Bar Chart ────────────────────────────────────────────────

def plot_alarm_vs_drop(df: pd.DataFrame, dataset_name: str) -> None:
    """Save a bar chart of alarm alpha vs. accuracy-drop alpha per failure mode.

    Args:
        df: Full detection_delay_table DataFrame.
        dataset_name: 'pima' or 'cleveland'.
    """
    subset  = df[df["dataset"] == dataset_name].copy()
    methods = list(METHOD_LABELS.keys())
    failure_modes = list(FAILURE_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = ["#e07070", "#e09050", "#5090d0", "#50b080"]

    for ax, fm in zip(axes, failure_modes):
        fm_data = subset[subset["failure_mode"] == fm]
        x = np.arange(len(methods))

        alarms, drops = [], []
        for m in methods:
            rows = fm_data[fm_data["method"] == m]
            if rows.empty:
                alarms.append(np.nan)
                drops.append(np.nan)
            else:
                row = rows.iloc[0]
                alarms.append(row["alarm_alpha"] if pd.notna(row["alarm_alpha"]) else np.nan)
                drops.append(row["drop_alpha"]   if pd.notna(row["drop_alpha"])  else np.nan)

        drop_val = drops[0] if drops and pd.notna(drops[0]) else np.nan

        for i, (m, alarm) in enumerate(zip(methods, alarms)):
            label = METHOD_LABELS[m] if fm == failure_modes[0] else ""
            ax.bar(x[i] - 0.15, alarm, width=0.28, color=colors[i], label=label)

        ax.bar(x + 0.15, [drop_val] * len(methods), width=0.28,
               label="Acc Drop alpha" if fm == failure_modes[0] else "",
               color="#7090d0", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split()[0] for m in methods], fontsize=8)
        ax.set_title(FAILURE_LABELS[fm], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Severity alpha" if ax == axes[0] else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"Alarm vs Accuracy-Drop Severity -- {dataset_name.capitalize()} Dataset",
        fontsize=12, y=1.06,
    )
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, f"alarm_vs_drop_{dataset_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Training Curves (combined) ─────────────────────────────────────────────

def plot_training_curves(losses_path: str | None = None) -> None:
    """Save combined MLP vs TabTransformer training loss curves for all datasets.

    Args:
        losses_path: Path to training_losses.json. Defaults to results/.
    """
    if losses_path is None:
        losses_path = os.path.join(RESULTS_PATH, "training_losses.json")
    if not os.path.exists(losses_path):
        print(f"  Skipping training curves: {losses_path} not found.")
        return

    with open(losses_path) as f:
        all_losses = json.load(f)

    datasets = list(all_losses.keys())
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    model_styles = {
        "mlp":            {"color": "#4a90d9", "label": "MC Dropout MLP",  "lw": 1.8},
        "tabtransformer": {"color": "#e07070", "label": "TabTransformer",   "lw": 1.8},
    }

    for ax, ds in zip(axes, datasets):
        for key, style in model_styles.items():
            if key not in all_losses[ds]:
                continue
            raw = all_losses[ds][key]
            window   = max(1, len(raw) // 30)
            smoothed = np.convolve(raw, np.ones(window) / window, mode="valid")
            ax.plot(np.linspace(1, len(raw), len(smoothed)), smoothed,
                    color=style["color"], lw=style["lw"], label=style["label"])
            ax.plot(range(1, len(raw) + 1), raw,
                    color=style["color"], alpha=0.2, lw=0.8)
        ax.set_title(f"{ds.capitalize()} Dataset -- Training Loss", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Deep Learning Training Curves: MLP vs TabTransformer", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Training Curves (per dataset) ─────────────────────────────────────────

def plot_training_curves_per_dataset(
    dataset_name: str,
    losses_path: str | None = None,
) -> None:
    """Save per-dataset training curves to results/training_curves_{dataset}.png.

    Args:
        dataset_name: 'pima' or 'cleveland'.
        losses_path: Path to training_losses.json. Defaults to results/.
    """
    if losses_path is None:
        losses_path = os.path.join(RESULTS_PATH, "training_losses.json")
    if not os.path.exists(losses_path):
        print(f"  Skipping per-dataset curves: {losses_path} not found.")
        return

    with open(losses_path) as f:
        all_losses = json.load(f)

    if dataset_name not in all_losses:
        print(f"  No loss data for '{dataset_name}' in {losses_path}.")
        return

    model_styles = {
        "mlp":            {"color": "#4a90d9", "label": "MC Dropout MLP",  "lw": 1.8},
        "tabtransformer": {"color": "#e07070", "label": "TabTransformer",   "lw": 1.8},
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for key, style in model_styles.items():
        if key not in all_losses[dataset_name]:
            continue
        raw      = all_losses[dataset_name][key]
        window   = max(1, len(raw) // 30)
        smoothed = np.convolve(raw, np.ones(window) / window, mode="valid")
        ax.plot(np.linspace(1, len(raw), len(smoothed)), smoothed,
                color=style["color"], lw=style["lw"], label=style["label"])
        ax.plot(range(1, len(raw) + 1), raw,
                color=style["color"], alpha=0.2, lw=0.8)

    ax.set_title(f"{dataset_name.capitalize()} -- Training Loss", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.suptitle("Deep Learning Training Curves: MLP vs TabTransformer", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_PATH, f"training_curves_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RESULTS_PATH, "detection_delay_table.csv"))
    for dataset_name in df["dataset"].unique():
        plot_detection_delay_heatmap(df, dataset_name)
        plot_alarm_vs_drop(df, dataset_name)

    losses_path = os.path.join(RESULTS_PATH, "training_losses.json")
    plot_training_curves(losses_path)
    if os.path.exists(losses_path):
        with open(losses_path) as f:
            ds_keys = list(json.load(f).keys())
        for ds in ds_keys:
            plot_training_curves_per_dataset(ds, losses_path)

    print("\nAll plots saved to results/")
