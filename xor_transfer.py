#Transfer Learning Attack on XOR Arbiter PUFs
# Question: does pretraining on PUF Instance A help attack Instance B with fewer CRPs?
# steps
#   1. Train a full MLP on Instance A (puf_seed=1) with 200k CRPs — the "source" PUF
#   2. Freeze the first 2 hidden layers, fine-tune only the last layer on Instance B
#      (puf_seed=99) using a small number of CRPs — the "target" PUF
#   3. Compare against MLP trained from scratch on Instance B with the same small CRP count
#   4. Repeat across multiple CRP counts: 1k, 2k, 5k, 10k, 25k, 50k


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
CHALLENGE_LENGTH = 64
K_XOR            = 4
SOURCE_SEED      = 1    # Instance A — train the pretrained model on this
TARGET_SEED      = 99   # Instance B — attack this with limited data
CHALLENGE_SEED_A = 2    # Challenges used for Instance A
CHALLENGE_SEED_B = 42   # Challenges used for Instance B (different set)
SOURCE_CRPS      = 500000

# How many CRPs to try for fine-tuning / scratch training on Instance B
TARGET_CRP_COUNTS = [1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000, ]
TEST_CRPS         = 40000  # Fixed test set for Instance B

EPOCHS_PRETRAIN   = 100   # Epochs for source model
EPOCHS_FINETUNE   = 50    # Epochs for fine-tuning
EPOCHS_SCRATCH    = 100   # Epochs for scratch model

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
# Generic train function
# -----------------------------
def train(model, X_tr, y_tr, X_v, y_v, X_te, y_te,
          epochs=100, lr=1e-3, batch_size=256, desc=""):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

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
                y_v.numpy(), (torch.sigmoid(model(X_v)) > 0.5).numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_acc = accuracy_score(
            y_te.numpy(), (torch.sigmoid(model(X_te)) > 0.5).numpy())

    return best_val_acc, test_acc

# -----------------------------
# Build tensors from numpy arrays
# -----------------------------
def to_tensors(X_tr, y_tr, X_v, y_v, X_te, y_te, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
    else:
        X_tr_s = scaler.transform(X_tr)
    X_v_s  = scaler.transform(X_v)
    X_te_s = scaler.transform(X_te)
    return (
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr,   dtype=torch.float32),
        torch.tensor(X_v_s,  dtype=torch.float32),
        torch.tensor(y_v,    dtype=torch.float32),
        torch.tensor(X_te_s, dtype=torch.float32),
        torch.tensor(y_te,   dtype=torch.float32),
        scaler
    )

# -----------------------------
# STEP 1 — Generate Instance A data and train source model
# -----------------------------
print("=" * 60)
print("STEP 1 — Pretrain on Instance A (source PUF)")
print(f"  k={K_XOR}, seed={SOURCE_SEED}, {SOURCE_CRPS:,} CRPs")
print("=" * 60)

puf_a   = XORArbiterPUF(n=CHALLENGE_LENGTH, k=K_XOR, seed=SOURCE_SEED,  noisiness=0.0)
X_raw_a = random_inputs(n=CHALLENGE_LENGTH, N=SOURCE_CRPS, seed=CHALLENGE_SEED_A)
y_a     = (puf_a.eval(X_raw_a) == 1).astype(np.int64)
X_phi_a = arbiter_feature_transform(X_raw_a)

idx_a = np.arange(SOURCE_CRPS)
tr_a, tmp_a = train_test_split(idx_a, test_size=0.30, random_state=42, stratify=y_a)
v_a, te_a   = train_test_split(tmp_a, test_size=0.67, random_state=42, stratify=y_a[tmp_a])

Xtr_a, ytr_a = X_phi_a[tr_a], y_a[tr_a]
Xv_a,  yv_a  = X_phi_a[v_a],  y_a[v_a]
Xte_a, yte_a = X_phi_a[te_a], y_a[te_a]

Xtr_a_t, ytr_a_t, Xv_a_t, yv_a_t, Xte_a_t, yte_a_t, scaler_a = \
    to_tensors(Xtr_a, ytr_a, Xv_a, yv_a, Xte_a, yte_a)

source_model = PUF_MLP()
t0 = time.time()
val_src, test_src = train(source_model,
    Xtr_a_t, ytr_a_t, Xv_a_t, yv_a_t, Xte_a_t, yte_a_t,
    epochs=EPOCHS_PRETRAIN, desc="Source")
print(f"Source model — Val: {val_src:.4f}  Test: {test_src:.4f}  ({time.time()-t0:.0f}s)")

# Save source weights
torch.save(source_model.state_dict(), "source_model.pt")
print("Source weights saved to source_model.pt\n")

# -----------------------------
# STEP 2 — Generate Instance B data (large pool, fixed test set)
# -----------------------------
print("=" * 60)
print("STEP 2 — Generate Instance B (target PUF)")
print(f"  k={K_XOR}, seed={TARGET_SEED}")
print("=" * 60)

MAX_B   = max(TARGET_CRP_COUNTS) + TEST_CRPS
puf_b   = XORArbiterPUF(n=CHALLENGE_LENGTH, k=K_XOR, seed=TARGET_SEED, noisiness=0.0)
X_raw_b = random_inputs(n=CHALLENGE_LENGTH, N=MAX_B, seed=CHALLENGE_SEED_B)
y_b     = (puf_b.eval(X_raw_b) == 1).astype(np.int64)
X_phi_b = arbiter_feature_transform(X_raw_b)

# Fix the test set as the LAST TEST_CRPS samples — same for every experiment
X_te_b  = X_phi_b[-TEST_CRPS:]
y_te_b  = y_b[-TEST_CRPS:]

# Scale test set using source scaler (same feature space)
X_te_b_t = torch.tensor(scaler_a.transform(X_te_b), dtype=torch.float32)
y_te_b_t = torch.tensor(y_te_b, dtype=torch.float32)

print(f"Instance B pool: {MAX_B:,} CRPs  |  Fixed test set: {TEST_CRPS:,} CRPs\n")

# -----------------------------
# STEP 3 — Run experiments across CRP counts
# -----------------------------
print("=" * 60)
print("STEP 3 — Fine-tune vs Scratch across CRP counts")
print("=" * 60)
print(f"\n{'CRPs':<10} {'Scratch val':>12} {'Scratch test':>13} {'Transfer val':>13} {'Transfer test':>14} {'Gain':>8}")
print("-" * 74)

scratch_results  = []
transfer_results = []

for n in TARGET_CRP_COUNTS:
    # Slice n CRPs for training (from the pool, excluding test set)
    X_pool = X_phi_b[:max(TARGET_CRP_COUNTS)]
    y_pool = y_b[:max(TARGET_CRP_COUNTS)]

    idx_pool = np.arange(len(X_pool))
    tr_b, tmp_b = train_test_split(idx_pool, test_size=0.25,
                                    random_state=42, stratify=y_pool)
    # Take only n training samples
    tr_b_n = tr_b[:n]
    v_b    = tmp_b

    Xtr_b  = X_pool[tr_b_n]
    ytr_b  = y_pool[tr_b_n]
    Xv_b   = X_pool[v_b]
    yv_b   = y_pool[v_b]

    # Scale using source scaler for consistency
    Xtr_b_t = torch.tensor(scaler_a.transform(Xtr_b), dtype=torch.float32)
    ytr_b_t = torch.tensor(ytr_b, dtype=torch.float32)
    Xv_b_t  = torch.tensor(scaler_a.transform(Xv_b),  dtype=torch.float32)
    yv_b_t  = torch.tensor(yv_b,  dtype=torch.float32)

    # --- Scratch model ---
    scratch = PUF_MLP()
    v_sc, t_sc = train(scratch,
        Xtr_b_t, ytr_b_t, Xv_b_t, yv_b_t, X_te_b_t, y_te_b_t,
        epochs=EPOCHS_SCRATCH, lr=1e-3)

    # --- Transfer model ---
    # Load source weights, freeze first 2 hidden layers (indices 0-5 in net)
    transfer = PUF_MLP()
    transfer.load_state_dict(torch.load("source_model.pt"))

    # Freeze: net[0]=Linear, net[1]=ReLU, net[2]=Dropout (layer 1)
    #         net[3]=Linear, net[4]=ReLU, net[5]=Dropout (layer 2)
    # Unfreeze: net[6]=Linear, net[7]=ReLU, net[8]=Dropout (layer 3)
    #           net[9]=Linear (output)
    for i, layer in enumerate(transfer.net):
        if i < 6:  # freeze first 2 hidden layers
            for param in layer.parameters() if hasattr(layer, 'parameters') else []:
                param.requires_grad = False
        # 
        #if i < 6:  # freeze first 2 hidden layers
            #for param in layer.parameters() if hasattr(layer, 'parameters') else []:
                #param.requires_grad = False

    v_tr, t_tr = train(transfer,
        Xtr_b_t, ytr_b_t, Xv_b_t, yv_b_t, X_te_b_t, y_te_b_t,
        epochs=EPOCHS_FINETUNE, lr=5e-4)  # lower lr for fine-tuning

    gain = t_tr - t_sc
    scratch_results.append((v_sc, t_sc))
    transfer_results.append((v_tr, t_tr))

    marker = " ← transfer wins" if gain > 0.01 else (" ← scratch wins" if gain < -0.01 else "")
    print(f"{n:<10,} {v_sc:>12.4f} {t_sc:>13.4f} {v_tr:>13.4f} {t_tr:>14.4f} {gain:>+8.4f}{marker}")

# -----------------------------
# STEP 4 — Summary
# -----------------------------
print(f"\n{'='*60}")
print("SUMMARY — Transfer Learning vs Scratch (k=4)")
print(f"{'='*60}")
print(f"Source PUF:  seed={SOURCE_SEED}  ({SOURCE_CRPS:,} CRPs to pretrain)")
print(f"Target PUF:  seed={TARGET_SEED}  (limited CRPs for fine-tune/scratch)")
print()
print(f"{'CRPs':<10} {'Scratch':>10} {'Transfer':>11} {'Gain':>8}")
print("-" * 42)
for i, n in enumerate(TARGET_CRP_COUNTS):
    v_sc, t_sc = scratch_results[i]
    v_tr, t_tr = transfer_results[i]
    gain = t_tr - t_sc
    marker = " ✓" if gain > 0.01 else ""
    print(f"{n:<10,} {t_sc:>10.4f} {t_tr:>11.4f} {gain:>+8.4f}{marker}")

print()
print("Source model on Instance A (for reference):")
print(f"  Val: {val_src:.4f}  Test: {test_src:.4f}")

# Save results
np.savez("day4_transfer_results.npz",
    target_crp_counts=np.array(TARGET_CRP_COUNTS),
    scratch_test=np.array([r[1] for r in scratch_results]),
    transfer_test=np.array([r[1] for r in transfer_results]),
    source_test=np.array([test_src]),
)
print("\nResults saved to day4_transfer_results.npz")