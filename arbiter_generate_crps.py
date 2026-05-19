

import numpy as np
from sklearn.model_selection import train_test_split
from pypuf.simulation import ArbiterPUF
from pypuf.io import random_inputs


CHALLENGE_LENGTH = 64
NUM_CRPS = 10000
PUF_SEED = 1
CHALLENGE_SEED = 2
NOISINESS = 0.0

TEST_SIZE = 0.20
VAL_SIZE = 0.10


def arbiter_feature_transform(challenges: np.ndarray) -> np.ndarray:

    return np.cumprod(challenges[:, ::-1], axis=1)[:, ::-1]



puf = ArbiterPUF(
    n=CHALLENGE_LENGTH,
    seed=PUF_SEED,
    noisiness=NOISINESS
)


X_raw = random_inputs(
    n=CHALLENGE_LENGTH,
    N=NUM_CRPS,
    seed=CHALLENGE_SEED
)



y_raw = puf.eval(X_raw)


y = (y_raw == 1).astype(np.int64)



X_bits = (X_raw == 1).astype(np.int64)


X_raw_float = X_raw.astype(np.float32)

X_phi = arbiter_feature_transform(X_raw).astype(np.float32)



indices = np.arange(NUM_CRPS)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=(TEST_SIZE + VAL_SIZE),
    random_state=42,
    shuffle=True,
    stratify=y
)


val_fraction_of_temp = VAL_SIZE / (TEST_SIZE + VAL_SIZE)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=(1.0 - val_fraction_of_temp),
    random_state=42,
    shuffle=True,
    stratify=y[temp_idx]
)


output_name = f"arbiter_puf_{CHALLENGE_LENGTH}bit_{NUM_CRPS}crps.npz"

np.savez(
    output_name,

    challenge_length=CHALLENGE_LENGTH,
    num_crps=NUM_CRPS,
    puf_seed=PUF_SEED,
    challenge_seed=CHALLENGE_SEED,
    noisiness=NOISINESS,

    X_raw=X_raw,
    y_raw=y_raw,
    y=y,

    X_bits_train=X_bits[train_idx],
    X_bits_val=X_bits[val_idx],
    X_bits_test=X_bits[test_idx],

    X_raw_train=X_raw_float[train_idx],
    X_raw_val=X_raw_float[val_idx],
    X_raw_test=X_raw_float[test_idx],

    X_phi_train=X_phi[train_idx],
    X_phi_val=X_phi[val_idx],
    X_phi_test=X_phi[test_idx],

    y_train=y[train_idx],
    y_val=y[val_idx],
    y_test=y[test_idx],
)



print(f"Saved: {output_name}")
print()
print("Dataset Summary")

print(f"Challenge length : {CHALLENGE_LENGTH}")
print(f"Number of CRPs   : {NUM_CRPS}")
print(f"Noise            : {NOISINESS}")
print(f"PUF type         : Arbiter PUF")
print()
print(f"Train samples    : {len(train_idx)}")
print(f"Val samples      : {len(val_idx)}")
print(f"Test samples     : {len(test_idx)}")
print()
print("Shapes:")
print("X_raw_train:", X_raw_float[train_idx].shape)
print("X_phi_train:", X_phi[train_idx].shape)
print("y_train    :", y[train_idx].shape)
print()
unique, counts = np.unique(y, return_counts=True)
print("Class balance:")
print(dict(zip(unique, counts)))