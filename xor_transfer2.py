# DAY 4C — Systematic freezing experiment
# Tests four transfer strategies vs scratch across CRP counts:
#   - Scratch:        random init, all layers train
#   - Warm start:     pretrained init, all layers train (freeze 0)
#   - Freeze 1 layer: pretrained init, first hidden layer frozen
#   - Freeze 2 layers: pretrained init, first 2 hidden layers frozen (original)
#
# Usage: python day4c_freezing.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
import time

# -----------------------------
# Configuration
# -----------------------------
CHALLENGE_LENGTH  = 64
K_XOR             = 4
SOURCE_SEED       = 1
TARGET_SEED       = 99
CHALLENGE_SEED_A  = 2
CHALLENGE_SEED_B  = 42
SOURCE_CRPS       = 500000
TARGET_CRP_COUNTS = [5000, 10000, 25000, 50000, 100000]
TEST_CRPS         = 40000
EPOCHS_SOURCE     = 100
EPOCHS_TARGET     = 100

# -----------------------------
# Feature transform
# -----------------------------
def arbiter_feature_transform(challenges):
    return np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1].astype(np.float32)

# -----------------------------
# MLP
# -----------------------------
class PUF_MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128,64,32], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# net layout:                            

def freeze_layers(model, n_hidden_layers_to_freeze):
    """Freeze the first n hidden layers (3 sublayers each: Linear+ReLU+Dropout)."""
    cutoff = n_hidden_layers_to_freeze * 3
    for i, module in enumerate(model.net):
        if i < cutoff:
            for param in module.parameters() if hasattr(module, 'parameters') else []:
                param.requires_grad = False
        else:
            for param in module.parameters() if hasattr(module, 'parameters') else []:
                param.requires_grad = True

# -----------------------------
# Train and return test accuracy
# -----------------------------
def train_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                epochs=100, lr=1e-3, batch_size=256):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy_score(
                y_v.numpy(), (torch.sigmoid(model(X_v)) > 0.5).numpy())

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = accuracy_score(
            y_te.numpy(), (torch.sigmoid(model(X_te)) > 0.5).numpy())
    return test_acc

# -----------------------------
# STEP 1 — Load or train source model
# -----------------------------
print("=" * 62)
print("STEP 1 — Source model on Instance A")
print("=" * 62)

import os
if os.path.exists("source_model.pt"):
    print("Found source_model.pt — loading existing weights.")
    source_state = torch.load("source_model.pt")
else:
    print(f"Training source model on {SOURCE_CRPS:,} CRPs (k={K_XOR}, seed={SOURCE_SEED})...")
    puf_a   = XORArbiterPUF(n=CHALLENGE_LENGTH, k=K_XOR, seed=SOURCE_SEED, noisiness=0.0)
    X_raw_a = random_inputs(n=CHALLENGE_LENGTH, N=SOURCE_CRPS, seed=CHALLENGE_SEED_A)
    y_a     = (puf_a.eval(X_raw_a) == 1).astype(np.int64)
    X_phi_a = arbiter_feature_transform(X_raw_a)

    idx = np.arange(SOURCE_CRPS)
    tr, tmp = train_test_split(idx, test_size=0.30, random_state=42, stratify=y_a)
    v, te   = train_test_split(tmp, test_size=0.67, random_state=42, stratify=y_a[tmp])

    sc = StandardScaler()
    Xtr = torch.tensor(sc.fit_transform(X_phi_a[tr]), dtype=torch.float32)
    Xv  = torch.tensor(sc.transform(X_phi_a[v]),      dtype=torch.float32)
    Xte = torch.tensor(sc.transform(X_phi_a[te]),     dtype=torch.float32)
    ytr = torch.tensor(y_a[tr], dtype=torch.float32)
    yv  = torch.tensor(y_a[v],  dtype=torch.float32)
    yte = torch.tensor(y_a[te], dtype=torch.float32)

    src = PUF_MLP()
    t0  = time.time()
    acc = train_model(src, Xtr, ytr, Xv, yv, Xte, yte, epochs=EPOCHS_SOURCE)
    print(f"Source model test accuracy: {acc:.4f}  ({time.time()-t0:.0f}s)")
    torch.save(src.state_dict(), "source_model.pt")
    source_state = src.state_dict()

print()

# -----------------------------
# STEP 2 — Generate Instance B
# -----------------------------
print("=" * 62)
print("STEP 2 — Instance B (target PUF, seed=99)")
print("=" * 62)
MAX_B   = max(TARGET_CRP_COUNTS) + TEST_CRPS
puf_b   = XORArbiterPUF(n=CHALLENGE_LENGTH, k=K_XOR, seed=TARGET_SEED, noisiness=0.0)
X_raw_b = random_inputs(n=CHALLENGE_LENGTH, N=MAX_B, seed=CHALLENGE_SEED_B)
y_b     = (puf_b.eval(X_raw_b) == 1).astype(np.int64)
X_phi_b = arbiter_feature_transform(X_raw_b)

X_te_np = X_phi_b[-TEST_CRPS:]
y_te_np = y_b[-TEST_CRPS:]
print(f"Pool: {MAX_B:,}  |  Fixed test set: {TEST_CRPS:,}\n")

# -----------------------------
# STEP 3 — Four strategies across CRP counts
# -----------------------------
STRATEGIES = {
    "Scratch":     {"freeze": None,  "lr": 1e-3},
    "Warm start":  {"freeze": 0,     "lr": 1e-3},
    "Freeze 1":    {"freeze": 1,     "lr": 5e-4},
    "Freeze 2":    {"freeze": 2,     "lr": 5e-4},
}

results = {s: [] for s in STRATEGIES}

print("=" * 62)
print("STEP 3 — Comparing strategies")
print("=" * 62)

col = 14
print(f"\n{'CRPs':<10}" + "".join(f"{s:>{col}}" for s in STRATEGIES))
print("-" * (10 + col * len(STRATEGIES)))

for n in TARGET_CRP_COUNTS:
    X_pool = X_phi_b[:max(TARGET_CRP_COUNTS)]
    y_pool = y_b[:max(TARGET_CRP_COUNTS)]

    idx = np.arange(len(X_pool))
    tr_idx, tmp_idx = train_test_split(idx, test_size=0.20,
                                        random_state=42, stratify=y_pool)
    tr_idx_n = tr_idx[:n]

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_pool[tr_idx_n]), dtype=torch.float32)
    X_v  = torch.tensor(scaler.transform(X_pool[tmp_idx]),      dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_te_np),              dtype=torch.float32)
    y_tr = torch.tensor(y_pool[tr_idx_n], dtype=torch.float32)
    y_v  = torch.tensor(y_pool[tmp_idx],  dtype=torch.float32)
    y_te = torch.tensor(y_te_np,          dtype=torch.float32)

    print(f"{n:<10,}", end="", flush=True)

    for strategy, cfg in STRATEGIES.items():
        model = PUF_MLP()

        # Load pretrained weights for all transfer strategies
        if cfg["freeze"] is not None:
            model.load_state_dict(source_state)
            freeze_layers(model, cfg["freeze"])
        # else: scratch — random init, all layers trainable

        acc = train_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                          epochs=EPOCHS_TARGET, lr=cfg["lr"])
        results[strategy].append(acc)
        print(f"{acc:>{col}.4f}", end="", flush=True)

    print()

# -----------------------------
# Summary
# -----------------------------
print(f"\n{'='*62}")
print("SUMMARY — Freezing strategy comparison (k=4, Instance B)")
print(f"{'CRPs':<10}" + "".join(f"{s:>{col}}" for s in STRATEGIES))
print("-" * (10 + col * len(STRATEGIES)))
for i, n in enumerate(TARGET_CRP_COUNTS):
    print(f"{n:<10,}", end="")
    best_acc = max(results[s][i] for s in STRATEGIES)
    for s in STRATEGIES:
        acc = results[s][i]
        marker = " ★" if acc == best_acc and len(STRATEGIES) > 1 else ""
        print(f"{acc:>{col}.4f}{marker[:2] if acc==best_acc else '  '}", end="")
    print()

print(f"\n★ = best strategy at that CRP count")

np.savez("day4c_freezing_results.npz",
    crp_counts=np.array(TARGET_CRP_COUNTS),
    **{s.replace(" ","_"): np.array(v) for s, v in results.items()})
print("Saved to day4c_freezing_results.npz")