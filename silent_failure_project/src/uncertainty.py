import numpy as np
from mapie.classification import SplitConformalClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from scipy.stats import entropy as scipy_entropy
from src.models import mc_dropout_predict, tabtransformer_predict


# ─────────────────────────────────────────────
# 1. Conformal Prediction (MAPIE)
# ─────────────────────────────────────────────

def fit_conformal(base_model, X_train, y_train, X_cal, y_cal, alpha=0.1):
    """
    Wraps a fitted XGBoost model with MAPIE conformal prediction.
    Returns the conformalized SplitConformalClassifier.
    MAPIE 1.x API: pass prefit estimator, then call conformalize().
    """
    mapie = SplitConformalClassifier(
        estimator=base_model,
        conformity_score="lac",
        prefit=True,
    )
    mapie.conformalize(X_cal, y_cal)
    return mapie


def conformal_uncertainty(mapie_model, X, alpha=0.1):
    """
    Returns uncertainty as 1 - max softmax probability (non-conformity score proxy).
    Higher = more uncertain.
    """
    proba = mapie_model._estimator.predict_proba(X)  # shape (N, 2)
    uncertainty = 1.0 - proba.max(axis=1)
    return uncertainty


# ─────────────────────────────────────────────
# 2. NGBoost (predictive distribution entropy)
# ─────────────────────────────────────────────

def fit_ngboost(X_train, y_train, n_estimators=100, random_state=42):
    ngb = NGBClassifier(
        Dist=Bernoulli,
        n_estimators=n_estimators,
        verbose=False,
        random_state=random_state,
    )
    ngb.fit(X_train, y_train)
    return ngb


def ngboost_uncertainty(ngb_model, X):
    """
    Returns entropy of the predicted Bernoulli distribution.
    H(p) = -p*log(p) - (1-p)*log(1-p)
    """
    proba = ngb_model.predict_proba(X)  # shape (N, 2)
    p = proba[:, 1].clip(1e-6, 1 - 1e-6)
    unc = scipy_entropy([p, 1 - p], base=2).T
    return unc


# ─────────────────────────────────────────────
# 3. MC Dropout MLP Variance  (DL baseline #1)
# ─────────────────────────────────────────────

def mcdropout_uncertainty(mlp_model, X, n_passes=50):
    """
    Returns epistemic uncertainty as variance across MC Dropout forward passes.
    """
    _, variance = mc_dropout_predict(mlp_model, X, n_passes=n_passes)
    return variance


# ─────────────────────────────────────────────
# 4. TabTransformer uncertainty  (DL baseline #2)
# ─────────────────────────────────────────────

def tabtransformer_uncertainty(tabt_model, X, n_passes=50):
    """
    Returns uncertainty as binary entropy of the mean predicted probability,
    averaged over n_passes stochastic forward passes (dropout active).

    H(p) = -p*log(p) - (1-p)*log(1-p)

    Using entropy of the mean (not raw MC variance) is more reliable for
    Transformer architectures because residual connections + LayerNorm suppress
    MC variance structurally — the entropy of the mean probability provides a
    cleaner signal of prediction confidence and shifts detectably with
    covariate drift.
    """
    mean_prob, _ = tabtransformer_predict(tabt_model, X, n_passes=n_passes)
    p = mean_prob.clip(1e-6, 1 - 1e-6)
    entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return entropy


# ─────────────────────────────────────────────
# Unified interface
# ─────────────────────────────────────────────

METHODS = ["conformal", "ngboost", "mcdropout", "tabtransformer"]


def get_uncertainty(method, models_dict, X, n_passes=50):
    """
    Unified call. models_dict must contain the relevant fitted model:
      "mapie"          -> SplitConformalClassifier (conformal)
      "ngboost"        -> NGBClassifier
      "mlp"            -> MCDropoutMLP
      "tabtransformer" -> TabTransformer
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
        raise ValueError(f"Unknown method: {method}")
