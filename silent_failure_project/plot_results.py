"""
plot_results.py
Generates all figures for the report from results/detection_delay_table.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

METHOD_LABELS = {
    "conformal":  "Conformal (MAPIE)",
    "ngboost":    "NGBoost Entropy",
    "mcdropout":  "MC Dropout MLP",
}

FAILURE_LABELS = {
    "covariate_shift":    "Covariate Shift",
    "label_noise":        "Label Noise",
    "feature_missingness":"Feature Missingness",
}


def plot_detection_delay_heatmap(df, dataset_name):
    subset = df[df["dataset"] == dataset_name].copy()
    subset["method_label"]  = subset["method"].map(METHOD_LABELS)
    subset["failure_label"] = subset["failure_mode"].map(FAILURE_LABELS)

    pivot = subset.pivot(
        index="failure_label",
        columns="method_label",
        values="detection_delay"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Detection Delay (α_alarm − α_drop)\nNegative = early warning"},
    )
    ax.set_title(
        f"Detection Delay Heatmap — {dataset_name.capitalize()} Dataset\n"
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
    subset = df[df["dataset"] == dataset_name].copy()
    methods = list(METHOD_LABELS.keys())
    failure_modes = list(FAILURE_LABELS.keys())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, fm in zip(axes, failure_modes):
        fm_data = subset[subset["failure_mode"] == fm]
        x = np.arange(len(methods))
        alarms = []
        drops  = []

        for m in methods:
            row = fm_data[fm_data["method"] == m].iloc[0]
            alarms.append(row["alarm_alpha"] if row["alarm_alpha"] is not None else np.nan)
            drops.append(row["drop_alpha"]  if row["drop_alpha"]  is not None else np.nan)

        ax.bar(x - 0.18, alarms, width=0.35, label="Alarm α", color="#e07070")
        ax.bar(x + 0.18, [drops[0]] * len(methods), width=0.35, label="Acc Drop α", color="#7090d0", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m].split()[0] for m in methods], fontsize=9)
        ax.set_title(FAILURE_LABELS[fm], fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Severity α" if ax == axes[0] else "")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Alarm vs Accuracy-Drop Severity — {dataset_name.capitalize()} Dataset",
        fontsize=12,
    )
    plt.tight_layout()
    path = f"results/alarm_vs_drop_{dataset_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    df = pd.read_csv("results/detection_delay_table.csv")
    for dataset_name in df["dataset"].unique():
        plot_detection_delay_heatmap(df, dataset_name)
        plot_alarm_vs_drop(df, dataset_name)
    print("\n✓ All plots saved to results/")
