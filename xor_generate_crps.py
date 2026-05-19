# Generate XOR Arbiter PUF Challenge-Response Pairs
# Run: python xor_generate_crps.py
# Output: xor_arbiter_puf_64bit_4xor_50000crps.npz

import numpy as np
from sklearn.model_selection import train_test_split
from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs

# -----------------------------
# Configuration
# -----------------------------
CHALLENGE_LENGTH = 64
NUM_CRPS = 50000
K_XOR = 4
PUF_SEED = 1
CHALLENGE_SEED = 2
NOISINESS = 0.0
TEST_SIZE = 0.20
VAL_SIZE = 0.15

# -----------------------------
# Feature transform
# -----------------------------
def arbiter_feature_transform(challenges: np.ndarray) -> np.ndarray:
    return np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1]

# -----------------------------
# 1. Create PUF
# -----------------------------
puf = XORArbiterPUF(n=CHALLENGE_LENGTH, k=K_XOR, seed=PUF_SEED, noisiness=NOISINESS)

# -----------------------------
# 2. Generate challenges & responses
# -----------------------------
X_raw = random_inputs(n=CHALLENGE_LENGTH, N=NUM_CRPS, seed=CHALLENGE_SEED)
y_raw = puf.eval(X_raw)
y = (y_raw == 1).astype(np.int64)

# -----------------------------
# 3. Build input variants
# -----------------------------
X_bits = (X_raw == 1).astype(np.int64)
X_phi = arbiter_feature_transform(X_raw).astype(np.float32)
X_raw_float = X_raw.astype(np.float32)

# -----------------------------
# 4. Train/Val/Test split (70/10/20)
# -----------------------------
indices = np.arange(NUM_CRPS)
train_idx, temp_idx = train_test_split(indices, test_size=(TEST_SIZE + VAL_SIZE),
                                        random_state=42, shuffle=True, stratify=y)
val_frac = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
val_idx, test_idx = train_test_split(temp_idx, test_size=(1.0 - val_frac),
                                      random_state=42, shuffle=True, stratify=y[temp_idx])

# -----------------------------
# 5. Save
# -----------------------------
output_name = f"xor_arbiter_puf_{CHALLENGE_LENGTH}bit_{K_XOR}xor_{NUM_CRPS}crps.npz"
np.savez(output_name,
    challenge_length=CHALLENGE_LENGTH, num_crps=NUM_CRPS, k_xor=K_XOR,
    puf_seed=PUF_SEED, challenge_seed=CHALLENGE_SEED, noisiness=NOISINESS,
    X_raw=X_raw, y_raw=y_raw, y=y,
    X_bits_train=X_bits[train_idx], X_bits_val=X_bits[val_idx], X_bits_test=X_bits[test_idx],
    X_raw_train=X_raw_float[train_idx], X_raw_val=X_raw_float[val_idx], X_raw_test=X_raw_float[test_idx],
    X_phi_train=X_phi[train_idx], X_phi_val=X_phi[val_idx], X_phi_test=X_phi[test_idx],
    y_train=y[train_idx], y_val=y[val_idx], y_test=y[test_idx],
)

print(f"Saved: {output_name}")
print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
unique, counts = np.unique(y, return_counts=True)
print(f"Class balance: {dict(zip(unique, counts))}")
