"""
plot_results.py
Generates all figures from results/detection_delay_table.csv
and results/training_losses.json (written by run_experiments.py).
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

METHOD_LABELS = {
    "conformal":      "Conformal (MAPIE)",
    "ngboost":        "NGBoost Entropy",
    "mcdropout":      "MC Dropout MLP",
    "tabtransformer": "TabTransformer",
}

FAILURE_LABELS = {
    "covariate_shift":     "Covariate Shift",
    "label_noise":         "Label Noise",
    "feature_missingness": "Feature Missingness",
}


def plot_detection_delay_heatmap(df, dataset_name):
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
        f"Negative values = method warned before accuracy drop",
        fontsize=11,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    path = f"results/heatmap_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_alarm_vs_drop(df, dataset_name):
    subset  = df[df["dataset"] == dataset_name].copy()
    methods = list(METHOD_LABELS.keys())
    failure_modes = list(FAILURE_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    colors = ["#e07070", "#e09050", "#5090d0", "#50b080"]  # one per method

    for ax, fm in zip(axes, failure_modes):
        fm_data = subset[subset["failure_mode"] == fm]
        x       = np.arange(len(methods))
        alarms  = []
        drops   = []

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
            ax.bar(x[i] - 0.15, alarm,    width=0.28, color=colors[i], label=METHOD_LABELS[m] if fm == failure_modes[0] else "")
        ax.bar(x + 0.15, [drop_val] * len(methods), width=0.28,
               label="Acc Drop α" if fm == failure_modes[0] else "",
               color="#7090d0", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split()[0] for m in methods], fontsize=8)
        ax.set_title(FAILURE_LABELS[fm], fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Severity α" if ax == axes[0] else "")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"Alarm vs Accuracy-Drop Severity -- {dataset_name.capitalize()} Dataset",
        fontsize=12, y=1.06,
    )
    plt.tight_layout()
    path = f"results/alarm_vs_drop_{dataset_name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_training_curves(losses_path="results/training_losses.json"):
    """
    Plot training loss curves for MC Dropout MLP and TabTransformer on both datasets.
    Reads the JSON written by run_experiments.py.
    """
    if not os.path.exists(losses_path):
        print(f"  Skipping training curves: {losses_path} not found. Run run_experiments.py first.")
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
        for model_key, style in model_styles.items():
            if model_key not in all_losses[ds]:
                continue
            raw = all_losses[ds][model_key]
            # Smooth with a simple rolling window for readability
            window = max(1, len(raw) // 30)
            smoothed = np.convolve(raw, np.ones(window) / window, mode="valid")
            epochs_sm = np.linspace(1, len(raw), len(smoothed))
            ax.plot(epochs_sm, smoothed, color=style["color"],
                    lw=style["lw"], label=style["label"])
            ax.plot(range(1, len(raw) + 1), raw,
                    color=style["color"], alpha=0.2, lw=0.8)

        ax.set_title(f"{ds.capitalize()} Dataset -- Training Loss", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Deep Learning Training Curves: MLP vs TabTransformer", fontsize=13)
    plt.tight_layout()
    path = "results/training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    df = pd.read_csv("results/detection_delay_table.csv")
    for dataset_name in df["dataset"].unique():
        plot_detection_delay_heatmap(df, dataset_name)
        plot_alarm_vs_drop(df, dataset_name)
    plot_training_curves()
    print("\nAll plots saved to results/")
