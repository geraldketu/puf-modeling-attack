#  Vary k AND CRP count
# For each k in [1,2,3,4,5,6], tests CRP counts: 50k, 100k, 200k, 500k, 1M
# Runs LR on X_phi and MLP on X_phi for every combination
# Usage: python xor_vary_k.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs
import time

# -----------------------------
# Configuration
# -----------------------------
CHALLENGE_LENGTH = 64
PUF_SEED         = 1
CHALLENGE_SEED   = 2
K_VALUES         = [1, 2, 3, 4, 5, 6]
CRP_COUNTS       = [50000, 100000, 200000, 500000, 1000000]
TEST_SIZE        = 0.20
VAL_SIZE         = 0.10

# -----------------------------
# Feature transform
# -----------------------------
def arbiter_feature_transform(challenges):
    return np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1].astype(np.float32)

# -----------------------------
# MLP definition
# -----------------------------
class PUF_MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128, 64, 32], dropout=0.2):
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

# -----------------------------
# Generate full 1M pool once per k
# -----------------------------
def generate_pool(k, max_crps):
    print(f"  Generating {max_crps:,} CRPs for k={k}...", end=" ", flush=True)
    t0    = time.time()
    puf   = XORArbiterPUF(n=CHALLENGE_LENGTH, k=k, seed=PUF_SEED, noisiness=0.0)
    X_raw = random_inputs(n=CHALLENGE_LENGTH, N=max_crps, seed=CHALLENGE_SEED)
    y     = (puf.eval(X_raw) == 1).astype(np.int64)
    X_phi = arbiter_feature_transform(X_raw)
    print(f"done ({time.time()-t0:.1f}s)")
    return X_phi, y

def slice_dataset(X_phi, y, n_crps):
    X      = X_phi[:n_crps]
    labels = y[:n_crps]
    idx    = np.arange(n_crps)
    train_idx, temp_idx = train_test_split(
        idx, test_size=(TEST_SIZE + VAL_SIZE),
        random_state=42, shuffle=True, stratify=labels)
    val_frac = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1.0 - val_frac),
        random_state=42, shuffle=True, stratify=labels[temp_idx])
    return {
        "X_phi_train": X[train_idx], "X_phi_val": X[val_idx], "X_phi_test": X[test_idx],
        "y_train": labels[train_idx], "y_val": labels[val_idx], "y_test": labels[test_idx],
        "n_train": len(train_idx),
    }

# -----------------------------
# Train LR on X_phi
# -----------------------------
def run_lr(ds):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(ds["X_phi_train"])
    X_te = scaler.transform(ds["X_phi_test"])
    lr   = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_tr, ds["y_train"])
    return accuracy_score(ds["y_test"], lr.predict(X_te))

# -----------------------------
# Train MLP on X_phi
# -----------------------------
def run_mlp(ds, epochs=100):
    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(ds["X_phi_train"]), dtype=torch.float32)
    X_v  = torch.tensor(scaler.transform(ds["X_phi_val"]),       dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(ds["X_phi_test"]),      dtype=torch.float32)
    y_tr = torch.tensor(ds["y_train"], dtype=torch.float32)
    y_v  = torch.tensor(ds["y_val"],   dtype=torch.float32)
    y_te = torch.tensor(ds["y_test"],  dtype=torch.float32)

    model     = PUF_MLP(input_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    best_val_acc = 0.0
    best_state   = None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy_score(
                y_v.numpy(),
                (torch.sigmoid(model(X_v)) > 0.5).numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {key: val.clone() for key, val in model.state_dict().items()}

        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = accuracy_score(
            y_te.numpy(),
            (torch.sigmoid(model(X_te)) > 0.5).numpy())
    return test_acc

# -----------------------------
# Run all
# -----------------------------
result_grid = {}
total_start = time.time()

for k in K_VALUES:
    print(f"\n{'='*62}")
    print(f"  k = {k}  ({len(CRP_COUNTS)} CRP sizes to run)")
    print(f"{'='*62}")

    X_phi_pool, y_pool = generate_pool(k, max(CRP_COUNTS))
    result_grid[k] = {}

    print(f"  {'CRPs':<12} {'Train':>9} {'LR':>10} {'MLP':>10} {'LR_t':>7} {'MLP_t':>7}")
    print(f"  {'-'*60}")

    for n in CRP_COUNTS:
        ds = slice_dataset(X_phi_pool, y_pool, n)

        t0     = time.time()
        lr_acc = run_lr(ds)
        lr_t   = time.time() - t0

        t0      = time.time()
        mlp_acc = run_mlp(ds, epochs=100)
        mlp_t   = time.time() - t0

        result_grid[k][n] = (lr_acc, mlp_acc)

        print(f"  {n:<12,} {ds['n_train']:>9,} {lr_acc:>10.4f} {mlp_acc:>10.4f} "
              f"{lr_t:>6.1f}s {mlp_t:>6.1f}s")

# -----------------------------
# Print full summary grids
# -----------------------------
crp_labels = [f"{n//1000}k" for n in CRP_COUNTS]
col_w = 10

print(f"\n\n{'='*70}")
print("LR (X_phi) test accuracy — rows=k, cols=CRP count")
print(f"{'='*70}")
print(f"{'':6}" + "".join(f"{c:>{col_w}}" for c in crp_labels))
print("-" * (6 + col_w * len(CRP_COUNTS)))
for k in K_VALUES:
    row = f"k={k:<4}"
    for n in CRP_COUNTS:
        lr_acc, _ = result_grid[k][n]
        row += f"{lr_acc:>{col_w}.4f}"
    print(row)

print(f"\n{'='*70}")
print("MLP (X_phi) test accuracy — rows=k, cols=CRP count")
print(f"{'='*70}")
print(f"{'':6}" + "".join(f"{c:>{col_w}}" for c in crp_labels))
print("-" * (6 + col_w * len(CRP_COUNTS)))
for k in K_VALUES:
    row = f"k={k:<4}"
    for n in CRP_COUNTS:
        _, mlp_acc = result_grid[k][n]
        row += f"{mlp_acc:>{col_w}.4f}"
    print(row)

elapsed = (time.time() - total_start) / 60
print(f"\nTotal runtime: {elapsed:.1f} minutes")

# -----------------------------
# Save results
lr_grid  = np.array([[result_grid[k][n][0] for n in CRP_COUNTS] for k in K_VALUES])
mlp_grid = np.array([[result_grid[k][n][1] for n in CRP_COUNTS] for k in K_VALUES])
np.savez("xor_combined_results.npz",
    k_values=np.array(K_VALUES),
    crp_counts=np.array(CRP_COUNTS),
    lr_results=lr_grid,
    mlp_results=mlp_grid)
print("Saved to xor_combined_results.npz")

