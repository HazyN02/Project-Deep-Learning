import numpy as np
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score


# ─────────────────────────────────────────────
# XGBoost Baseline
# ─────────────────────────────────────────────

def get_xgboost(random_state=42):
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


def train_xgboost(X_train, y_train, random_state=42):
    model = get_xgboost(random_state)
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"[XGBoost] 5-fold CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    return model


# ─────────────────────────────────────────────
# MC Dropout MLP  (deep learning baseline #1)
# ─────────────────────────────────────────────

class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def enable_dropout(self):
        """Keep dropout active at inference time for MC sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_mlp(X_train, y_train, input_dim, epochs=150, lr=1e-3,
              hidden_dim=64, dropout=0.3, random_state=42):
    """
    Returns (model, loss_history).
    """
    torch.manual_seed(random_state)
    model = MCDropoutMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 30 == 0:
            print(f"  [MLP] Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return model, losses


def mc_dropout_predict(model, X, n_passes=50):
    """
    Run n_passes stochastic forward passes with dropout enabled.
    Returns:
        mean_probs: shape (N,)  — mean predicted probability
        variance:   shape (N,)  — epistemic uncertainty proxy
    """
    model.eval()
    model.enable_dropout()

    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    preds = np.stack(preds, axis=0)   # (n_passes, N)
    mean_probs = preds.mean(axis=0)
    variance   = preds.var(axis=0)
    return mean_probs, variance


# ─────────────────────────────────────────────
# TabTransformer  (deep learning baseline #2)
# ─────────────────────────────────────────────

class TabTransformer(nn.Module):
    """
    Lightweight TabTransformer for binary tabular classification.

    Architecture:
      1. Per-feature linear projection: (B, F) -> (B, F, embed_dim)
      2. num_layers x TransformerEncoderLayer (nhead, dim_feedforward, dropout)
      3. Global average pooling across the feature dimension: (B, embed_dim)
      4. Two-layer MLP head with dropout for binary classification.

    MC-Dropout uncertainty: call enable_dropout() before inference to keep
    attention + MLP dropout active, then run multiple stochastic forward passes.
    """

    def __init__(self, input_dim, embed_dim=32, nhead=4, num_layers=2,
                 dim_feedforward=64, attn_dropout=0.1, mlp_dropout=0.3):
        super().__init__()
        self.input_dim = input_dim

        # Shared linear projection: each feature scalar -> embed_dim vector
        self.feature_proj = nn.Linear(1, embed_dim)

        # Positional bias (learned, one per feature) so the transformer can
        # distinguish features even though inputs are permutation-invariant scalars
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            batch_first=True,
            norm_first=True,      # pre-norm for stability on small datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Two-layer MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        # x: (B, F)
        x = x.unsqueeze(-1)                   # (B, F, 1)
        x = self.feature_proj(x)              # (B, F, embed_dim)
        x = x + self.pos_embedding            # add learned positional bias
        x = self.transformer(x)               # (B, F, embed_dim)
        x = x.mean(dim=1)                     # global avg pool -> (B, embed_dim)
        return self.head(x).squeeze(-1)        # (B,)

    def enable_dropout(self):
        """Activate all Dropout layers at inference time for MC sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_tabtransformer(X_train, y_train, input_dim, epochs=250, lr=1e-3,
                         embed_dim=64, nhead=2, num_layers=2,
                         dim_feedforward=128, attn_dropout=0.3, mlp_dropout=0.5,
                         random_state=42):
    """
    Train a TabTransformer with BCEWithLogitsLoss + Adam + ReduceLROnPlateau.
    Returns (model, loss_history).
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=20, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())
        losses.append(loss.item())
        if (epoch + 1) % 40 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [TabTransformer] Epoch {epoch+1}/{epochs}  "
                  f"loss={loss.item():.4f}  lr={lr_now:.2e}")

    return model, losses


def tabtransformer_predict(model, X, n_passes=50):
    """
    Run n_passes stochastic forward passes with dropout enabled (MC sampling).
    Returns:
        mean_probs: shape (N,)  — mean predicted probability
        variance:   shape (N,)  — epistemic uncertainty proxy
    """
    model.eval()
    model.enable_dropout()

    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)

    preds = np.stack(preds, axis=0)   # (n_passes, N)
    mean_probs = preds.mean(axis=0)
    variance   = preds.var(axis=0)
    return mean_probs, variance
