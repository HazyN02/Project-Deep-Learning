"""
src/models.py
Model definitions, training functions, and wrapper classes.

All four uncertainty estimators share a common interface:
  .fit(X_train, y_train)
  .predict_proba(X)           → np.ndarray shape (n, 2)
  .predict_uncertainty(X)     → np.ndarray shape (n,), higher = more uncertain
  .score(X, y)                → float accuracy

Training functions return (model, loss_history) for DL models,
or (model, None) for classical models, for uniform handling.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.config import (
    RANDOM_SEED,
    XGB_PARAMS,
    NGB_N_ESTIMATORS,
    MLP_HIDDEN_DIM, MLP_DROPOUT, MLP_EPOCHS, MLP_LR,
    TABT_EMBED_DIM, TABT_NHEAD, TABT_NUM_LAYERS, TABT_DIM_FEEDFORWARD,
    TABT_ATTN_DROPOUT, TABT_MLP_DROPOUT, TABT_EPOCHS, TABT_LR,
    TABT_WEIGHT_DECAY, TABT_LR_PATIENCE, TABT_LR_FACTOR, TABT_MIN_LR,
    TABT_GRAD_CLIP,
)


# ── XGBoost ────────────────────────────────────────────────────────────────

def get_xgboost(random_state: int = RANDOM_SEED) -> XGBClassifier:
    """Instantiate an XGBoost classifier with project-standard params.

    Args:
        random_state: Seed passed to XGBClassifier.

    Returns:
        Unfitted XGBClassifier.
    """
    params = dict(XGB_PARAMS)
    params["random_state"] = random_state
    return XGBClassifier(**params)


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> XGBClassifier:
    """Fit an XGBoost classifier and print 5-fold CV AUC.

    Args:
        X_train: Training features, shape (n, d).
        y_train: Training labels, shape (n,).
        random_state: Seed for reproducibility.

    Returns:
        Fitted XGBClassifier.
    """
    model = get_xgboost(random_state)
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  [XGBoost] 5-fold CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return model


# ── MC Dropout MLP ─────────────────────────────────────────────────────────

class MCDropoutMLP(nn.Module):
    """Two-hidden-layer MLP with dropout at inference time for MC sampling.

    Architecture: Linear(d->H) -> ReLU -> Dropout(p) ->
                  Linear(H->H) -> ReLU -> Dropout(p) ->
                  Linear(H->1)

    MC Dropout: call enable_dropout() before inference to keep Dropout layers
    in train mode, then run multiple forward passes to estimate epistemic
    uncertainty via prediction variance.
    """

    def __init__(self, input_dim: int, hidden_dim: int = MLP_HIDDEN_DIM,
                 dropout: float = MLP_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def enable_dropout(self) -> None:
        """Set all Dropout layers to train mode (activates stochastic masking)."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
    hidden_dim: int = MLP_HIDDEN_DIM,
    dropout: float = MLP_DROPOUT,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """Train an MCDropoutMLP with BCE loss and Adam optimizer.

    Args:
        X_train: Training features, shape (n, d).
        y_train: Training labels, shape (n,).
        input_dim: Number of input features.
        epochs: Training epochs.
        lr: Learning rate.
        hidden_dim: Hidden layer width.
        dropout: Dropout probability.
        random_state: Torch manual seed.

    Returns:
        (model, losses): Trained model and per-epoch BCE loss history.
    """
    torch.manual_seed(random_state)
    model     = MCDropoutMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if (epoch + 1) % 30 == 0:
            print(f"    [MLP] Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return model, losses


def mc_dropout_predict(
    model: MCDropoutMLP,
    X: np.ndarray,
    n_passes: int = 50,
) -> tuple:
    """Run n_passes stochastic forward passes with dropout active.

    Args:
        model: Trained MCDropoutMLP.
        X: Feature array, shape (n, d).
        n_passes: Number of stochastic forward passes.

    Returns:
        mean_probs: Mean predicted probability per sample, shape (n,).
        variance: Epistemic uncertainty proxy (MC variance), shape (n,).
    """
    model.eval()
    model.enable_dropout()

    X_t   = torch.tensor(X, dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(torch.sigmoid(model(X_t)).cpu().numpy())

    preds      = np.stack(preds, axis=0)   # (n_passes, n)
    mean_probs = preds.mean(axis=0)
    variance   = preds.var(axis=0)
    return mean_probs, variance


# ── TabTransformer ─────────────────────────────────────────────────────────

class TabTransformer(nn.Module):
    """Lightweight TabTransformer for binary tabular classification.

    Architecture:
      1. Per-feature linear projection: (B, F) -> (B, F, embed_dim)
      2. Learned positional bias: one embedding per feature position
      3. num_layers x TransformerEncoderLayer (pre-norm for stability)
      4. Global average pooling across feature dimension: (B, embed_dim)
      5. Two-layer MLP head with dropout -> scalar logit

    Uncertainty: call enable_dropout() before inference and run multiple
    stochastic forward passes. Use entropy of the MEAN predicted probability
    (not raw MC variance) as the uncertainty signal -- residual connections
    and LayerNorm suppress MC variance structurally in Transformer blocks.
    See tabtransformer_predict() and KEY FIX comment in uncertainty.py.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = TABT_EMBED_DIM,
        nhead: int = TABT_NHEAD,
        num_layers: int = TABT_NUM_LAYERS,
        dim_feedforward: int = TABT_DIM_FEEDFORWARD,
        attn_dropout: float = TABT_ATTN_DROPOUT,
        mlp_dropout: float = TABT_MLP_DROPOUT,
    ):
        super().__init__()
        self.input_dim    = input_dim
        self.feature_proj = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, input_dim, embed_dim) * 0.01
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,   # pre-norm for stability on small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)         # (B, F, 1)
        x = self.feature_proj(x)    # (B, F, E)
        x = x + self.pos_embedding  # add learned positional bias
        x = self.transformer(x)     # (B, F, E)
        x = x.mean(dim=1)           # global avg pool -> (B, E)
        return self.head(x).squeeze(-1)   # (B,)

    def enable_dropout(self) -> None:
        """Activate all Dropout layers for MC sampling at inference time."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_tabtransformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int = TABT_EPOCHS,
    lr: float = TABT_LR,
    embed_dim: int = TABT_EMBED_DIM,
    nhead: int = TABT_NHEAD,
    num_layers: int = TABT_NUM_LAYERS,
    dim_feedforward: int = TABT_DIM_FEEDFORWARD,
    attn_dropout: float = TABT_ATTN_DROPOUT,
    mlp_dropout: float = TABT_MLP_DROPOUT,
    random_state: int = RANDOM_SEED,
) -> tuple:
    """Train a TabTransformer with BCE loss, Adam, and ReduceLROnPlateau.

    Args:
        X_train: Training features, shape (n, d).
        y_train: Training labels, shape (n,).
        input_dim: Number of input features.
        epochs: Training epochs.
        lr: Initial learning rate.
        embed_dim: Feature embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of TransformerEncoderLayer blocks.
        dim_feedforward: FFN hidden size per Transformer block.
        attn_dropout: Dropout in attention layers.
        mlp_dropout: Dropout in MLP head.
        random_state: Torch manual seed.

    Returns:
        (model, losses): Trained model and per-epoch BCE loss history.
    """
    torch.manual_seed(random_state)
    model = TabTransformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        attn_dropout=attn_dropout,
        mlp_dropout=mlp_dropout,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=TABT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=TABT_LR_PATIENCE,
        factor=TABT_LR_FACTOR,
        min_lr=TABT_MIN_LR,
    )
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=TABT_GRAD_CLIP)
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(float(loss.item()))
        if (epoch + 1) % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    [TabTransformer] Epoch {epoch+1}/{epochs}  "
                  f"loss={loss.item():.4f}  lr={lr_now:.2e}")

    return model, losses


def tabtransformer_predict(
    model: TabTransformer,
    X: np.ndarray,
    n_passes: int = 50,
) -> tuple:
    """Run n_passes stochastic forward passes with dropout active.

    Args:
        model: Trained TabTransformer.
        X: Feature array, shape (n, d).
        n_passes: Number of stochastic forward passes.

    Returns:
        mean_probs: Mean predicted probability, shape (n,).
        variance: MC variance across passes (informational only), shape (n,).
    """
    model.eval()
    model.enable_dropout()

    X_t   = torch.tensor(X, dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(torch.sigmoid(model(X_t)).cpu().numpy())

    preds      = np.stack(preds, axis=0)   # (n_passes, n)
    mean_probs = preds.mean(axis=0)
    variance   = preds.var(axis=0)
    return mean_probs, variance
