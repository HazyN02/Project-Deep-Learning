"""
ui/app.py
Streamlit interface for the Silent Failure Detection Benchmark.
Run with: streamlit run ui/app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from src.data_loader import load_dataset
from src.models import train_xgboost, train_mlp
from src.inject import apply_failure, FAILURE_MODES, SEVERITY_LEVELS
from src.uncertainty import fit_conformal, fit_ngboost, get_uncertainty, METHODS
from src.alarm import run_ks_alarm, accuracy_drop_alpha, detection_delay
from sklearn.metrics import accuracy_score, roc_auc_score

st.set_page_config(page_title="Silent Failure Detector", layout="wide")
st.title("🔍 When Do Clinical Tabular Models Fail Silently?")
st.markdown(
    "**A deployment monitoring benchmark** — inject controlled failures and watch "
    "which uncertainty method detects them first."
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    dataset_name  = st.selectbox("Dataset",      ["pima", "cleveland"])
    failure_mode  = st.selectbox("Failure Mode", list(FAILURE_MODES.keys()))
    alpha         = st.slider("Severity α", min_value=0.0, max_value=0.9,
                               step=0.1, value=0.3)
    n_passes      = st.slider("MC Dropout passes", 10, 100, 50, step=10)
    run_button    = st.button("▶ Run", type="primary")

# ── Model caching ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models (one-time)…")
def load_and_train(dataset_name):
    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler = load_dataset(dataset_name)
    input_dim = X_train.shape[1]

    xgb   = train_xgboost(X_train, y_train)
    ngb   = fit_ngboost(X_train, y_train)
    mlp   = train_mlp(X_train, y_train, input_dim=input_dim, epochs=150)
    mapie = fit_conformal(xgb, X_train, y_train, X_cal, y_cal)

    models = {"mapie": mapie, "ngboost": ngb, "mlp": mlp}
    data   = (X_train, X_cal, X_test, y_train, y_cal, y_test)
    return models, data, xgb


if run_button:
    models, data, xgb = load_and_train(dataset_name)
    X_train, X_cal, X_test, y_train, y_cal, y_test = data

    # Baseline
    baseline_acc = accuracy_score(y_test, xgb.predict(X_test))
    baseline_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])
    baseline_unc = {m: get_uncertainty(m, models, X_test, n_passes=n_passes) for m in METHODS}

    # Injected
    X_corr, y_corr = apply_failure(X_test, y_test, failure_mode, alpha)
    corr_acc = accuracy_score(y_corr, xgb.predict(X_corr))
    corr_unc = {m: get_uncertainty(m, models, X_corr, n_passes=n_passes) for m in METHODS}

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.subheader("📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline Accuracy", f"{baseline_acc:.3f}")
    col2.metric("Corrupted Accuracy", f"{corr_acc:.3f}",
                delta=f"{corr_acc - baseline_acc:+.3f}")
    col3.metric("Baseline AUC", f"{baseline_auc:.3f}")

    # ── Alarm status per method ───────────────────────────────────────────────
    st.subheader("🚨 Uncertainty Method Alarms")
    method_cols = st.columns(3)

    for col, method in zip(method_cols, METHODS):
        stat, p = ks_2samp(baseline_unc[method], corr_unc[method])
        alarmed = p < 0.05
        col.markdown(f"**{method}**")
        if alarmed:
            col.error(f"🔴 ALARM  (p={p:.4f})")
        else:
            col.success(f"🟢 OK  (p={p:.4f})")
        col.caption(f"KS stat: {stat:.4f}")

    # ── Uncertainty distributions ─────────────────────────────────────────────
    st.subheader("📈 Uncertainty Distributions (Clean vs Corrupted)")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    METHOD_LABELS = {
        "conformal": "Conformal (MAPIE)",
        "ngboost":   "NGBoost Entropy",
        "mcdropout": "MC Dropout MLP",
    }

    for ax, method in zip(axes, METHODS):
        ax.hist(baseline_unc[method], bins=20, alpha=0.6, label="Clean", color="#4a90d9")
        ax.hist(corr_unc[method],     bins=20, alpha=0.6, label=f"α={alpha}", color="#e07070")
        ax.set_title(METHOD_LABELS[method], fontsize=10)
        ax.set_xlabel("Uncertainty Score")
        ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Full severity sweep ───────────────────────────────────────────────────
    st.subheader("🔁 Detection Delay Across All Severity Levels")

    severity_unc = {m: [] for m in METHODS}
    y_trues, y_preds = [], []

    for sev in SEVERITY_LEVELS:
        Xc, yc = apply_failure(X_test, y_test, failure_mode, sev)
        y_trues.append(yc)
        y_preds.append(xgb.predict(Xc))
        for m in METHODS:
            severity_unc[m].append(get_uncertainty(m, models, Xc, n_passes=n_passes))

    drop_alpha, accs = accuracy_drop_alpha(y_trues, y_preds, baseline_acc)

    rows = []
    for method in METHODS:
        alarm_alpha, _ = run_ks_alarm(baseline_unc[method], severity_unc[method])
        delay = detection_delay(alarm_alpha, drop_alpha)
        rows.append({
            "Method":          METHOD_LABELS[method],
            "Alarm α":         alarm_alpha,
            "Acc Drop α":      drop_alpha,
            "Detection Delay": delay,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Accuracy curve
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    sev_vals = [a[0] for a in accs]
    acc_vals = [a[1] for a in accs]
    ax2.plot(sev_vals, acc_vals, marker="o", color="#4a90d9", label="Accuracy")
    ax2.axhline(baseline_acc, linestyle="--", color="gray", label="Baseline")
    if drop_alpha is not None:
        ax2.axvline(drop_alpha, linestyle=":", color="red", label=f"Drop α={drop_alpha}")
    ax2.set_xlabel("Severity α")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy Under {failure_mode.replace('_',' ').title()}")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

else:
    st.info("👈 Configure options in the sidebar and click **Run** to start.")
    st.markdown("""
    ### How it works
    1. **Select a dataset** — Pima Diabetes or Cleveland Heart Disease
    2. **Choose a failure mode** — covariate shift, label noise, or feature missingness
    3. **Set severity α** — how aggressively to corrupt the test set
    4. **Run** — see which uncertainty method raises the alarm earliest

    ### Uncertainty Methods
    | Method | Approach |
    |--------|----------|
    | **Conformal (MAPIE)** | Coverage-guaranteed prediction sets via split conformal |
    | **NGBoost Entropy** | Full predictive distribution, uncertainty via binary entropy |
    | **MC Dropout MLP** | Deep neural network with Bayesian dropout at inference time |
    """)
