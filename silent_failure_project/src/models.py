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
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )


def train_xgboost(X_train, y_train, random_state=42):
    model = get_xgboost(random_state)
    model.fit(X_train, y_train)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"[XGBoost] 5-fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return model


# ─────────────────────────────────────────────
# MC Dropout MLP
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
        """Keep dropout active at inference time."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()


def train_mlp(X_train, y_train, input_dim, epochs=100, lr=1e-3,
              hidden_dim=64, dropout=0.3, random_state=42):
    torch.manual_seed(random_state)
    model = MCDropoutMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = criterion(logits, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"  [MLP] Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")

    return model


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

    preds = np.stack(preds, axis=0)  # (n_passes, N)
    mean_probs = preds.mean(axis=0)
    variance   = preds.var(axis=0)
    return mean_probs, variance
