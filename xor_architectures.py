# Comparison: MLP vs LSTM vs CNN vs Transformer
# Tests four neural network architectures on k=4 XOR Arbiter PUF
# All trained on X_phi features
# Usage: python xor_architectures.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
PUF_SEED          = 1
CHALLENGE_SEED    = 2
CRP_COUNTS        = [10000, 25000, 50000]
TEST_CRPS         = 40000
EPOCHS            = 100
BATCH_SIZE        = 256

# -----------------------------
# Feature transform
# -----------------------------
def arbiter_feature_transform(challenges):
    return np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1].astype(np.float32)

# ================================================================
# ARCHITECTURE 1 
class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128,64,32], dropout=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 64)
        return self.net(x).squeeze(1)

# ================================================================
# ARCHITECTURE 2 — LSTM
# Reads challenge bits one at a time as a sequence of length 64.
# Each bit is one time step. The hidden state accumulates delay info.
# Physical motivation: signal travels through 64 arbiter stages in order.
# ================================================================
class LSTM_PUF(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 64) → reshape to (batch, 64, 1) for LSTM
        x = x.unsqueeze(2)
        out, (h_n, _) = self.lstm(x)
        # Use the last hidden state from the final layer
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.classifier(last_hidden).squeeze(1)

# ================================================================
# ARCHITECTURE 3 — CNN
class CNN_PUF(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        # Input: (batch, 1, 64) — 1 channel, length 64
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1,  out_channels=32, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),  # Pool to fixed size regardless of input length
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, 64) → (batch, 1, 64) for Conv1d
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x).squeeze(1)

# ================================================================
# ARCHITECTURE 4 — Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer_PUF(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        # Project each scalar bit to d_model dimensions
        self.embed   = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier  = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 64) → (batch, 64, 1) → embed → (batch, 64, d_model)
        x = self.embed(x.unsqueeze(2))
        x = self.pos_enc(x)
        x = self.transformer(x)
        # Aggregate across sequence with mean pooling
        x = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(x).squeeze(1)

# ================================================================
# Shared training function
# ================================================================
def train_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                epochs=100, lr=1e-3, batch_size=256):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    loader    = DataLoader(TensorDataset(X_tr, y_tr),
                           batch_size=batch_size, shuffle=True)

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
                y_v.numpy(),
                (torch.sigmoid(model(X_v)) > 0.5).numpy())

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = accuracy_score(
            y_te.numpy(),
            (torch.sigmoid(model(X_te)) > 0.5).numpy())
    return best_val, test_acc

# ================================================================
# Generate data
# ================================================================
print("Generating 1M CRP pool (k=4, seed=1)...")
t0    = time.time()
MAX   = max(CRP_COUNTS) + TEST_CRPS
puf   = XORArbiterPUF(n=CHALLENGE_LENGTH, k=K_XOR, seed=PUF_SEED, noisiness=0.0)
X_raw = random_inputs(n=CHALLENGE_LENGTH, N=MAX, seed=CHALLENGE_SEED)
y_all = (puf.eval(X_raw) == 1).astype(np.int64)
X_phi = arbiter_feature_transform(X_raw)

X_te_np = X_phi[-TEST_CRPS:]
y_te_np = y_all[-TEST_CRPS:]
print(f"Done ({time.time()-t0:.1f}s). Test set: {TEST_CRPS:,}\n")

# Architecture definitions
ARCHITECTURES = {
    "MLP":         lambda: MLP(),
    "LSTM":        lambda: LSTM_PUF(),
    "CNN":         lambda: CNN_PUF(),
    "Transformer": lambda: Transformer_PUF(),
}

# Print param counts
print("Parameter counts:")
for name, factory in ARCHITECTURES.items():
    m = factory()
    params = sum(p.numel() for p in m.parameters())
    print(f"  {name:<14} {params:>8,} parameters")
print()

# ================================================================
results = {name: {"val": [], "test": [], "time": []} for name in ARCHITECTURES}

col = 16
print(f"{'CRPs':<10}", end="")
for name in ARCHITECTURES:
    print(f"{name:>{col}}", end="")
print()
print("-" * (10 + col * len(ARCHITECTURES)))

for n in CRP_COUNTS:
    X_pool = X_phi[:-TEST_CRPS]
    y_pool = y_all[:-TEST_CRPS]

    idx = np.arange(len(X_pool))
    tr_idx, tmp_idx = train_test_split(
        idx, test_size=0.20, random_state=42, stratify=y_pool)
    tr_idx_n = tr_idx[:n]

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_pool[tr_idx_n])
    X_v_np  = scaler.transform(X_pool[tmp_idx])
    X_te_sc = scaler.transform(X_te_np)

    X_tr = torch.tensor(X_tr_np,  dtype=torch.float32)
    X_v  = torch.tensor(X_v_np,   dtype=torch.float32)
    X_te = torch.tensor(X_te_sc,  dtype=torch.float32)
    y_tr = torch.tensor(y_pool[tr_idx_n], dtype=torch.float32)
    y_v  = torch.tensor(y_pool[tmp_idx],  dtype=torch.float32)
    y_te = torch.tensor(y_te_np,          dtype=torch.float32)

    print(f"{n:<10,}", end="", flush=True)

    for name, factory in ARCHITECTURES.items():
        t0    = time.time()
        model = factory()
        bv, ta = train_model(model, X_tr, y_tr, X_v, y_v, X_te, y_te,
                              epochs=EPOCHS, lr=1e-3)
        elapsed = time.time() - t0
        results[name]["val"].append(bv)
        results[name]["test"].append(ta)
        results[name]["time"].append(elapsed)
        print(f"{ta:>{col}.4f}", end="", flush=True)

    print()

# ================================================================
# Summary

print(f"SUMMARY XOR PUF, X_phi features")
print(f"{'CRPs':<10}", end="")
for name in ARCHITECTURES:
    print(f"{name:>{col}}", end="")
print()
print("-" * (10 + col * len(ARCHITECTURES)))

for i, n in enumerate(CRP_COUNTS):
    print(f"{n:<10,}", end="")
    best = max(results[name]["test"][i] for name in ARCHITECTURES)
    for name in ARCHITECTURES:
        acc = results[name]["test"][i]
        star = "★" if acc == best else " "
        print(f"{acc:>{col-1}.4f}{star}", end="")
    print()

print(f"\n{'='*65}")
print("Average training time per CRP count:")
for name in ARCHITECTURES:
    avg_t = np.mean(results[name]["time"])
    print(f"  {name:<14} {avg_t:>6.1f}s per run")

print(f"\n★ = best architecture at that CRP count")

np.savez("day5_architecture_results.npz",
    crp_counts=np.array(CRP_COUNTS),
    **{f"{name}_test": np.array(results[name]["test"]) for name in ARCHITECTURES},
    **{f"{name}_val":  np.array(results[name]["val"])  for name in ARCHITECTURES},
)
print("Saved to xor_architecture_results.npz")
