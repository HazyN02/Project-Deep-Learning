"""
src/uncertainty.py
Uncertainty estimation wrappers for all four methods.

All methods expose a single entry point:
    get_uncertainty(method, models_dict, X, n_passes) -> np.ndarray shape (n,)
Higher values indicate higher uncertainty.
"""

import numpy as np
from mapie.classification import SplitConformalClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli

from src.config import METHODS, N_MC_PASSES, RANDOM_SEED
from src.models import mc_dropout_predict, tabtransformer_predict


# ── 1. Conformal Prediction (MAPIE) ───────────────────────────────────────

def fit_conformal(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    alpha: float = 0.1,
):
    """Wrap a fitted XGBoost model with MAPIE split conformal prediction.

    Uses the MAPIE 1.x API: SplitConformalClassifier with .conformalize().
    The base_model must already be fitted on X_train/y_train.

    Args:
        base_model: Fitted XGBClassifier (prefit=True).
        X_train: Training features (passed for API compatibility, not used).
        y_train: Training labels (passed for API compatibility, not used).
        X_cal: Calibration features for conformalization.
        y_cal: Calibration labels for conformalization.
        alpha: Miscoverage rate (default 0.1).

    Returns:
        Fitted SplitConformalClassifier.
    """
    mapie = SplitConformalClassifier(
        estimator=base_model,
        conformity_score="lac",
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)
    return mapie


def conformal_uncertainty(mapie_model, X: np.ndarray) -> np.ndarray:
    """Compute uncertainty as 1 - max class probability.

    Uses the public .estimator_ attribute (MAPIE 1.x) rather than the
    private ._estimator to avoid breakage across minor versions.

    Args:
        mapie_model: Fitted SplitConformalClassifier.
        X: Feature array, shape (n, d).

    Returns:
        Uncertainty scores, shape (n,). Higher = more uncertain.
    """
    # Public API (.estimator_) preferred; fall back to private (._estimator)
    # for compatibility with older MAPIE minor versions
    estimator = getattr(mapie_model, "estimator_", None) or mapie_model._estimator
    proba = estimator.predict_proba(X)   # shape (n, 2)
    return 1.0 - proba.max(axis=1)


# ── 2. NGBoost ────────────────────────────────────────────────────────────

def fit_ngboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    random_state: int = RANDOM_SEED,
) -> NGBClassifier:
    """Fit an NGBoost classifier with Bernoulli distribution.

    Args:
        X_train: Training features, shape (n, d).
        y_train: Training labels, shape (n,).
        n_estimators: Number of boosting rounds.
        random_state: Seed for reproducibility.

    Returns:
        Fitted NGBClassifier.
    """
    ngb = NGBClassifier(
        Dist=Bernoulli,
        n_estimators=n_estimators,
        verbose=False,
        random_state=random_state,
    )
    ngb.fit(X_train, y_train)
    return ngb


def ngboost_uncertainty(ngb_model, X: np.ndarray) -> np.ndarray:
    """Compute binary entropy of the NGBoost Bernoulli distribution.

    H(p) = -p*log2(p) - (1-p)*log2(1-p)

    Explicit formula used (not scipy.stats.entropy with array stacking)
    to avoid ambiguity when n=1 or when scipy API changes.

    Args:
        ngb_model: Fitted NGBClassifier.
        X: Feature array, shape (n, d).

    Returns:
        Entropy scores, shape (n,). Range [0, 1] bits.
    """
    proba = ngb_model.predict_proba(X)   # shape (n, 2)
    p     = proba[:, 1].clip(1e-6, 1 - 1e-6)
    entropy = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return entropy


# ── 3. MC Dropout MLP ─────────────────────────────────────────────────────

def mcdropout_uncertainty(mlp_model, X: np.ndarray, n_passes: int = 50) -> np.ndarray:
    """Epistemic uncertainty as variance across MC Dropout forward passes.

    Args:
        mlp_model: Trained MCDropoutMLP.
        X: Feature array, shape (n, d).
        n_passes: Number of stochastic forward passes.

    Returns:
        Variance scores, shape (n,). Higher = more uncertain.
    """
    _, variance = mc_dropout_predict(mlp_model, X, n_passes=n_passes)
    return variance


# ── 4. TabTransformer ─────────────────────────────────────────────────────

def tabtransformer_uncertainty(tabt_model, X: np.ndarray, n_passes: int = 50) -> np.ndarray:
    """Uncertainty as binary entropy of the mean predicted probability.

    KEY FIX: Using entropy of the mean prediction (not raw MC variance).
    Residual connections + LayerNorm in Transformer blocks structurally
    suppress MC variance -- dropout noise averages out across layers,
    making variance an unreliable shift signal for Transformer architectures.
    Entropy of the mean probability shifts detectably under covariate drift
    while remaining stable on clean data.

    See project report Section III for full analysis.

    Implementation:
        probs = [sigmoid(model(X)) for _ in range(n_passes)]
        mean_prob = stack(probs).mean(dim=0)
        uncertainty = -mean_prob*log(mean_prob) - (1-mean_prob)*log(1-mean_prob)

    Args:
        tabt_model: Trained TabTransformer.
        X: Feature array, shape (n, d).
        n_passes: Number of stochastic forward passes for mean estimation.

    Returns:
        Binary entropy scores, shape (n,). Higher = more uncertain.
    """
    mean_prob, _ = tabtransformer_predict(tabt_model, X, n_passes=n_passes)
    p = mean_prob.clip(1e-6, 1 - 1e-6)
    entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
    return entropy


# ── Unified interface ──────────────────────────────────────────────────────

def get_uncertainty(
    method: str,
    models_dict: dict,
    X: np.ndarray,
    n_passes: int = 50,
) -> np.ndarray:
    """Unified uncertainty call for all four methods.

    Args:
        method: One of 'conformal', 'ngboost', 'mcdropout', 'tabtransformer'.
        models_dict: Must contain keys:
            'mapie'          -> SplitConformalClassifier
            'ngboost'        -> NGBClassifier
            'mlp'            -> MCDropoutMLP
            'tabtransformer' -> TabTransformer
        X: Feature array, shape (n, d).
        n_passes: MC passes for DL methods (ignored by classical methods).

    Returns:
        Uncertainty scores, shape (n,).

    Raises:
        ValueError: If method is unknown.
        KeyError: If required model key is missing from models_dict.
    """
    if method == "conformal":
        return conformal_uncertainty(models_dict["mapie"], X)
    elif method == "ngboost":
        return ngboost_uncertainty(models_dict["ngboost"], X)
    elif method == "mcdropout":
        return mcdropout_uncertainty(models_dict["mlp"], X, n_passes=n_passes)
    elif method == "tabtransformer":
        return tabtransformer_uncertainty(models_dict["tabtransformer"], X, n_passes=n_passes)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: {METHODS}")
