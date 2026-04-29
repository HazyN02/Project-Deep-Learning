"""
src/config.py
Central configuration for the Silent Failure Detection pipeline.

All magic numbers, paths, and hyperparameters live here.
Every other module imports from this file — nothing is redefined elsewhere.
"""

import os

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data")
DOCS_PATH    = os.path.join(PROJECT_ROOT, "docs")

# ── Datasets ───────────────────────────────────────────────────────────────
DATASETS = ["pima", "cleveland"]

# ── Failure injection ──────────────────────────────────────────────────────
SEVERITY_LEVELS = [round(i * 0.1, 1) for i in range(10)]   # 0.0 … 0.9

# ── Uncertainty methods ────────────────────────────────────────────────────
METHODS = ["conformal", "ngboost", "mcdropout", "tabtransformer"]

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

# ── KS-test alarm ──────────────────────────────────────────────────────────
ALARM_P_THRESHOLD  = 0.05   # reject H0 if p < this
ALARM_CONSECUTIVE  = 3      # need this many consecutive rejections to fire
ACCURACY_DROP_THRESHOLD = 0.05   # fraction below baseline to define "drop"

# ── MC sampling ────────────────────────────────────────────────────────────
# Passes per method for stable uncertainty estimates
N_MC_PASSES = {
    "conformal":      1,
    "ngboost":        1,
    "mcdropout":      100,
    "tabtransformer": 100,
}

# ── XGBoost ────────────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_SEED,
    verbosity=0,
)

# ── NGBoost ────────────────────────────────────────────────────────────────
NGB_N_ESTIMATORS = 100

# ── MC Dropout MLP ─────────────────────────────────────────────────────────
MLP_HIDDEN_DIM = 64
MLP_DROPOUT    = 0.3
MLP_EPOCHS     = 150
MLP_LR         = 1e-3

# ── TabTransformer ─────────────────────────────────────────────────────────
TABT_EMBED_DIM       = 64
TABT_NHEAD           = 2
TABT_NUM_LAYERS      = 2
TABT_DIM_FEEDFORWARD = 128
TABT_ATTN_DROPOUT    = 0.3
TABT_MLP_DROPOUT     = 0.5
TABT_EPOCHS          = 250
TABT_LR              = 1e-3
TABT_WEIGHT_DECAY    = 1e-4
TABT_LR_PATIENCE     = 20
TABT_LR_FACTOR       = 0.5
TABT_MIN_LR          = 1e-5
TABT_GRAD_CLIP       = 1.0
