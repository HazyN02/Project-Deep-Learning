"""
ui/app.py
Streamlit interface for the Silent Failure Detection Benchmark.

Loads pre-computed results from results/*.csv by default.
Click "Rerun Evaluation" in the sidebar to retrain and refresh.

Run with: streamlit run ui/app.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score

from src.config import (
    RESULTS_PATH, METHODS, METHOD_LABELS, FAILURE_LABELS,
    SEVERITY_LEVELS, N_MC_PASSES, RANDOM_SEED,
)
from src.seed_everything import seed_everything

seed_everything(RANDOM_SEED)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Silent Failure Detector — Clinical ML Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Silent Failure Detector — Clinical ML Monitor")
st.markdown(
    "Monitors prediction uncertainty across four methods as an early-warning "
    "system for silent model degradation. Results loaded from pre-computed "
    "sweep; click **Rerun Evaluation** to refresh."
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    dataset_name = st.selectbox("Dataset", ["pima", "cleveland"])
    failure_mode = st.selectbox(
        "Failure Mode",
        list(FAILURE_LABELS.keys()),
        format_func=lambda k: FAILURE_LABELS[k],
    )
    alarm_thresh = st.slider(
        "Alarm Threshold (uncertainty)",
        min_value=0.0, max_value=1.0, value=0.05, step=0.005, format="%.3f",
    )
    selected_methods = st.multiselect(
        "Methods to display",
        options=METHODS,
        default=METHODS,
        format_func=lambda k: METHOD_LABELS[k],
    )
    n_passes = st.slider("MC / TT stochastic passes", 10, 100, 50, step=10)

    st.divider()
    rerun_btn = st.button(
        "Rerun Evaluation", type="primary",
        help="Retrain all models and regenerate results/ (~5 min).",
    )

# ── Rerun pipeline (Step 5e) ───────────────────────────────────────────────
if rerun_btn:
    with st.spinner("Running full experiment sweep (several minutes)..."):
        import subprocess
        result = subprocess.run(
            [sys.executable, "run_experiments.py"],
            capture_output=True, text=True,
        )
    if result.returncode == 0:
        st.success("Evaluation complete. Results updated.")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    else:
        st.error("Evaluation failed.")
        st.code(result.stderr[-3000:])
        st.stop()


# ── Model caching ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models (one-time, ~2 min)...")
def load_and_train(ds_name: str):
    """Train all models for the selected dataset and cache weights."""
    from src.data_loader import load_dataset
    from src.models import train_xgboost, train_mlp, train_tabtransformer
    from src.uncertainty import fit_conformal, fit_ngboost

    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler = load_dataset(ds_name)
    xgb        = train_xgboost(X_train, y_train)
    ngb        = fit_ngboost(X_train, y_train)
    mlp, _     = train_mlp(X_train, y_train, input_dim=X_train.shape[1])
    tabt, _    = train_tabtransformer(X_train, y_train, input_dim=X_train.shape[1])
    mapie      = fit_conformal(xgb, X_train, y_train, X_cal, y_cal)
    models = {"mapie": mapie, "ngboost": ngb, "mlp": mlp, "tabtransformer": tabt}
    return models, (X_train, X_cal, X_test, y_train, y_cal, y_test), xgb


# ── Severity sweep (cached per config combination) ─────────────────────────
@st.cache_data(show_spinner="Computing uncertainty sweep...", ttl=300)
def compute_sweep(ds_name: str, fm: str, n_passes_val: int):
    """Return per-severity mean uncertainty and accuracy for all methods."""
    from src.uncertainty import get_uncertainty
    from src.inject import apply_failure

    models, data, xgb = load_and_train(ds_name)
    _, _, X_test, _, _, y_test = data

    baseline_unc = {
        m: get_uncertainty(m, models, X_test, n_passes=n_passes_val)
        for m in METHODS
    }
    baseline_acc = float(accuracy_score(y_test, xgb.predict(X_test)))

    sev_mean_unc = {m: [] for m in METHODS}
    sev_acc = []

    for alpha in SEVERITY_LEVELS:
        Xc, yc = apply_failure(X_test, y_test, fm, alpha)
        sev_acc.append(float(accuracy_score(yc, xgb.predict(Xc))))
        for m in METHODS:
            unc = get_uncertainty(m, models, Xc, n_passes=n_passes_val)
            sev_mean_unc[m].append(float(unc.mean()))

    return baseline_unc, baseline_acc, sev_mean_unc, sev_acc


# ── Load pre-computed CSV results ──────────────────────────────────────────
delay_csv   = os.path.join(RESULTS_PATH, "detection_delay_table.csv")
metrics_csv = os.path.join(RESULTS_PATH, "metrics_comparison.csv")
results_ok  = os.path.exists(delay_csv)

if results_ok:
    df_delay   = pd.read_csv(delay_csv)
    df_metrics = pd.read_csv(metrics_csv) if os.path.exists(metrics_csv) else None
else:
    df_delay, df_metrics = None, None
    st.warning(
        "No pre-computed results found. "
        "Click **Rerun Evaluation** in the sidebar to generate them."
    )

# ── Attempt live sweep ─────────────────────────────────────────────────────
live_ok = False
try:
    baseline_unc, baseline_acc, sev_mean_unc, sev_acc = compute_sweep(
        dataset_name, failure_mode, n_passes
    )
    live_ok = True
except Exception as _live_exc:
    pass

# ══════════════════════════════════════════════════════════════════════════
# PANEL 1 — Uncertainty Monitor (Plotly line chart)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Panel 1 — Uncertainty Monitor")

if live_ok:
    # Accuracy drop point
    acc_drop_alpha = None
    for alpha, acc in zip(SEVERITY_LEVELS, sev_acc):
        if acc < baseline_acc - 0.05:
            acc_drop_alpha = alpha
            break

    METHOD_COLORS = {
        "conformal":      "#e07070",
        "ngboost":        "#e09050",
        "mcdropout":      "#5090d0",
        "tabtransformer": "#50b080",
    }

    fig = go.Figure()

    for method in (selected_methods or METHODS):
        fig.add_trace(go.Scatter(
            x=SEVERITY_LEVELS,
            y=sev_mean_unc[method],
            mode="lines+markers",
            name=METHOD_LABELS[method],
            line=dict(color=METHOD_COLORS[method], width=2),
            marker=dict(size=6),
        ))

    # Alarm threshold horizontal line
    fig.add_hline(
        y=alarm_thresh,
        line_dash="dash", line_color="red", line_width=1.5,
        annotation_text=f"Alarm threshold ({alarm_thresh:.3f})",
        annotation_position="top right",
    )

    # Accuracy drop vertical line
    if acc_drop_alpha is not None:
        fig.add_vline(
            x=acc_drop_alpha,
            line_dash="dot", line_color="orange", line_width=2,
            annotation_text=f"Acc drop (a={acc_drop_alpha})",
            annotation_position="top left",
        )

        # Shade detection window
        method_alarm_alphas = []
        for method in (selected_methods or METHODS):
            for alpha, unc_val in zip(SEVERITY_LEVELS, sev_mean_unc[method]):
                if unc_val >= alarm_thresh:
                    method_alarm_alphas.append(alpha)
                    break

        if method_alarm_alphas:
            earliest = min(method_alarm_alphas)
            fig.add_vrect(
                x0=min(earliest, acc_drop_alpha),
                x1=max(earliest, acc_drop_alpha),
                fillcolor="rgba(255,200,100,0.15)",
                layer="below", line_width=0,
                annotation_text="Detection window",
                annotation_position="top left",
            )

    fig.update_layout(
        title=(
            f"Mean Uncertainty vs. Severity — "
            f"{dataset_name.capitalize()} / {FAILURE_LABELS[failure_mode]}"
        ),
        xaxis_title="Severity (alpha)",
        yaxis_title="Mean Uncertainty Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(t=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Accuracy curve below
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=SEVERITY_LEVELS, y=sev_acc,
        mode="lines+markers", name="Accuracy",
        line=dict(color="#4a90d9", width=2), marker=dict(size=6),
    ))
    fig_acc.add_hline(
        y=baseline_acc,
        line_dash="dash", line_color="gray",
        annotation_text=f"Baseline ({baseline_acc:.3f})",
        annotation_position="right",
    )
    if acc_drop_alpha is not None:
        fig_acc.add_vline(
            x=acc_drop_alpha,
            line_dash="dot", line_color="orange", line_width=2,
        )
    fig_acc.update_layout(
        title="Accuracy Under Failure",
        xaxis_title="Severity (alpha)", yaxis_title="Accuracy",
        height=250, margin=dict(t=50),
    )
    st.plotly_chart(fig_acc, use_container_width=True)

else:
    st.info("Live sweep not available — showing pre-computed results only.")

# ══════════════════════════════════════════════════════════════════════════
# PANEL 2 — Detection Summary Table
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Panel 2 — Detection Summary Table")

if results_ok and df_delay is not None:
    subset = df_delay[
        (df_delay["dataset"] == dataset_name) &
        (df_delay["failure_mode"] == failure_mode)
    ].copy()

    if not subset.empty:
        display_rows = []
        for _, row in subset.iterrows():
            delay = row["detection_delay"]
            alarm = row["alarm_alpha"]
            drop  = row["drop_alpha"]

            if pd.isna(delay):
                delay_str, tag = "None", "red"
            elif float(delay) <= 0.0:
                delay_str, tag = f"{float(delay):+.1f}", "green"
            elif float(delay) <= 0.2:
                delay_str, tag = f"{float(delay):+.1f}", "yellow"
            else:
                delay_str, tag = f"{float(delay):+.1f}", "red"

            display_rows.append({
                "Method":          METHOD_LABELS.get(row["method"], row["method"]),
                "Alarm alpha":     f"{float(alarm):.1f}" if pd.notna(alarm) else "—",
                "Drop alpha":      f"{float(drop):.1f}"  if pd.notna(drop)  else "—",
                "Detection Delay": delay_str,
            })

        display_df = pd.DataFrame(display_rows)

        def _style_delay(val):
            if val == "None":
                return "color:#cc2222; font-weight:bold"
            try:
                v = float(val)
            except (ValueError, TypeError):
                return ""
            if v <= 0.0:
                return "color:#228822; font-weight:bold"
            if v <= 0.2:
                return "color:#cc8800; font-weight:bold"
            return "color:#cc2222; font-weight:bold"

        st.dataframe(
            display_df.style.applymap(_style_delay, subset=["Detection Delay"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "🟢 delay ≤ 0: alarm fires before accuracy drop (early warning)  |  "
            "🟡 0 < delay ≤ 0.2: slightly late (≤2 steps)  |  "
            "🔴 delay > 0.2 or None: significantly late / never fired"
        )
    else:
        st.info("No pre-computed results for this dataset / failure mode.")
else:
    st.info("Run the evaluation to populate this table.")

# ══════════════════════════════════════════════════════════════════════════
# PANEL 3 — Model Status Badges
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Panel 3 — Model Status")

methods_to_show = selected_methods if selected_methods else METHODS

if live_ok:
    # Live status: get uncertainty at alpha=0.5
    from src.inject import apply_failure as _apply
    from src.uncertainty import get_uncertainty as _get_unc

    try:
        models, data, xgb = load_and_train(dataset_name)
        _, _, X_test, _, _, y_test = data
        Xc05, yc05 = _apply(X_test, y_test, failure_mode, 0.5)
        corr_acc_05 = float(accuracy_score(yc05, xgb.predict(Xc05)))

        cols = st.columns(len(methods_to_show))
        for col, method in zip(cols, methods_to_show):
            unc_arr  = _get_unc(method, models, Xc05, n_passes=n_passes)
            mean_unc = float(unc_arr.mean())
            ratio    = mean_unc / alarm_thresh if alarm_thresh > 0 else 0.0

            if mean_unc >= alarm_thresh:
                badge, badge_color = "🚨 ALARM", "#ff4444"
            elif ratio >= 0.9:
                badge, badge_color = "⚠️ WARNING", "#ff9900"
            else:
                badge, badge_color = "✅ MONITORING", "#22aa22"

            col.markdown(
                f"**{METHOD_LABELS[method]}**  \n"
                f"<span style='font-size:1.2em;color:{badge_color}'>{badge}</span>",
                unsafe_allow_html=True,
            )
            col.metric(
                "Uncertainty (a=0.5)", f"{mean_unc:.4f}",
                delta=f"vs threshold {alarm_thresh:.3f}",
                delta_color="inverse" if mean_unc >= alarm_thresh else "off",
            )
            col.caption(f"Corrupted ACC @ a=0.5: {corr_acc_05:.3f}")

    except Exception as exc:
        st.info(f"Live status unavailable: {exc}")

elif results_ok and df_delay is not None and not subset.empty:
    # Fallback to sweep CSV
    cols = st.columns(len(methods_to_show))
    for col, method in zip(cols, methods_to_show):
        row_data = subset[subset["method"] == method]
        if row_data.empty:
            continue
        row_data = row_data.iloc[0]
        alarm = row_data["alarm_alpha"]

        if pd.isna(alarm):
            badge, badge_color = "✅ MONITORING", "#22aa22"
            note = "No alarm in sweep"
        else:
            badge, badge_color = "🚨 ALARM", "#ff4444"
            note = f"Alarm fired at a={float(alarm):.1f}"

        col.markdown(
            f"**{METHOD_LABELS[method]}**  \n"
            f"<span style='font-size:1.2em;color:{badge_color}'>{badge}</span>",
            unsafe_allow_html=True,
        )
        col.caption(note)

# ══════════════════════════════════════════════════════════════════════════
# Calibration Metrics (from pre-computed results)
# ══════════════════════════════════════════════════════════════════════════
if df_metrics is not None:
    st.subheader("Calibration Metrics — XGBoost Production Model")
    m_sub = df_metrics[df_metrics["dataset"] == dataset_name].copy()
    if not m_sub.empty:
        col_map = {
            "method_label":     "Method",
            "baseline_acc":     "Baseline ACC",
            "baseline_auc":     "Baseline AUC",
            "mean_uncertainty": "Mean Unc (clean)",
            "ECE":              "ECE",
            "Brier_Score":      "Brier Score",
        }
        display_cols = [c for c in col_map if c in m_sub.columns]
        st.dataframe(
            m_sub[display_cols].rename(columns=col_map),
            use_container_width=True,
            hide_index=True,
        )

# ══════════════════════════════════════════════════════════════════════════
# How it works expander
# ══════════════════════════════════════════════════════════════════════════
with st.expander("How it works", expanded=False):
    st.markdown("""
    ### Detection delay benchmark

    1. **Select a dataset** — Pima Diabetes or Cleveland Heart Disease
    2. **Choose a failure mode** — covariate shift, label noise, or feature missingness
    3. **Set alarm threshold** — uncertainty level that triggers a monitoring alarm
    4. **Panel 1** — Plotly line chart: mean uncertainty vs. severity for each method.
       The red dashed line is the alarm threshold; the orange dotted line is where
       accuracy measurably drops. The shaded region is the **detection window**.
    5. **Panel 2** — Detection delay table from the pre-computed sweep.
       Green = early warning, Yellow = slightly late, Red = late or never.
    6. **Panel 3** — Live alarm status at alpha=0.5 for the selected failure mode.

    ### Uncertainty methods

    | Method | Type | Uncertainty Signal |
    |--------|------|--------------------|
    | Conformal (MAPIE) | Classical | 1 - max softmax probability |
    | NGBoost Entropy | Classical | Binary entropy H(p) |
    | MC Dropout MLP | Deep Learning | Variance across 100 forward passes |
    | TabTransformer | Deep Learning | Entropy of mean across 100 passes |

    > **Key finding:** TabTransformer uses entropy of the *mean* prediction (not
    > raw MC variance) because residual connections + LayerNorm structurally suppress
    > variance in Transformer architectures. See report Section III.
    """)
