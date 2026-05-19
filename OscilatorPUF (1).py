import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



CHALLENGE_LENGTH = 64
NUM_CRPS = 10000
NUM_ROS = 256

BASE_FREQUENCY = 100e6          # 100 MHz nominal frequency
FREQUENCY_STD = 2e6             # manufacturing variation, std = 2 MHz

NOISE_STD = 0.0                 # noise = 0 initially

PUF_SEED = 1
CHALLENGE_SEED = 2
TEST_SIZE = 0.20



def bits_to_int(bits):

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value



# RO-PUF response function

def ro_puf_eval(challenges, ro_frequencies, noise_std=0.0, seed=123):
    """
    Evaluate Ring Oscillator PUF responses.

    Input:
        challenges: shape (N, 64), values are 0/1
        ro_frequencies: shape (NUM_ROS,)
        noise_std: frequency noise std

    Output:
        responses: shape (N,), values are 0/1
        selected_i: first RO index
        selected_j: second RO index
    """

    rng = np.random.default_rng(seed)

    N = challenges.shape[0]
    num_ros = len(ro_frequencies)

    responses = np.zeros(N, dtype=np.int64)
    selected_i = np.zeros(N, dtype=np.int64)
    selected_j = np.zeros(N, dtype=np.int64)

    for sample_idx in range(N):
        c = challenges[sample_idx]

        # First 32 bits select RO_i
        left_bits = c[:32]

        # Last 32 bits select RO_j
        right_bits = c[32:]

        i = bits_to_int(left_bits) % num_ros
        j = bits_to_int(right_bits) % num_ros

        # Avoid comparing the same oscillator with itself
        if i == j:
            j = (j + 1) % num_ros

        # Add measurement noise if needed
        noise_i = rng.normal(0, noise_std)
        noise_j = rng.normal(0, noise_std)

        freq_i = ro_frequencies[i] + noise_i
        freq_j = ro_frequencies[j] + noise_j

        # Response bit
        if freq_i > freq_j:
            responses[sample_idx] = 1
        else:
            responses[sample_idx] = 0

        selected_i[sample_idx] = i
        selected_j[sample_idx] = j

    return responses, selected_i, selected_j


# 1. Create one RO-PUF instance

puf_rng = np.random.default_rng(PUF_SEED)

ro_frequencies = puf_rng.normal(
    loc=BASE_FREQUENCY,
    scale=FREQUENCY_STD,
    size=NUM_ROS
)



# 2. Generate random 64-bit challenges

challenge_rng = np.random.default_rng(CHALLENGE_SEED)

X_bits = challenge_rng.integers(
    low=0,
    high=2,
    size=(NUM_CRPS, CHALLENGE_LENGTH),
    dtype=np.int64
)



# 3. Evaluate RO-PUF to generate responses

y, ro_i, ro_j = ro_puf_eval(
    challenges=X_bits,
    ro_frequencies=ro_frequencies,
    noise_std=NOISE_STD,
    seed=3
)



# 4. Train/test split

indices = np.arange(NUM_CRPS)

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=42,
    shuffle=True,
    stratify=y
)

X_train = X_bits[train_idx]
X_test = X_bits[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

ro_i_train = ro_i[train_idx]
ro_i_test = ro_i[test_idx]

ro_j_train = ro_j[train_idx]
ro_j_test = ro_j[test_idx]



# 5. Prepare CNN input

# For PyTorch Conv1D later:
# shape should be (N, channels, length)
X_train_cnn = X_train.reshape(-1, 1, CHALLENGE_LENGTH)
X_test_cnn = X_test.reshape(-1, 1, CHALLENGE_LENGTH)



# 6. Save NPZ file for ML

np.savez(
    "ro_puf_64bit_10000crps.npz",
    X_train=X_train,
    X_test=X_test,
    X_train_cnn=X_train_cnn,
    X_test_cnn=X_test_cnn,
    y_train=y_train,
    y_test=y_test,
    ro_i_train=ro_i_train,
    ro_i_test=ro_i_test,
    ro_j_train=ro_j_train,
    ro_j_test=ro_j_test,
    ro_frequencies=ro_frequencies,
    X_all=X_bits,
    y_all=y,
    ro_i_all=ro_i,
    ro_j_all=ro_j
)



df = pd.DataFrame(
    X_bits,
    columns=[f"c{i}" for i in range(CHALLENGE_LENGTH)]
)

df["ro_i"] = ro_i
df["ro_j"] = ro_j
df["response"] = y

df.to_csv("ro_puf_64bit_10000crps.csv", index=False)




print("Challenge length:", CHALLENGE_LENGTH)
print("Number of CRPs:", NUM_CRPS)
print("Number of ring oscillators:", NUM_ROS)
print("Response length: 1 bit")
print("Noise std:", NOISE_STD)
print()
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)
print("X_train_cnn shape:", X_train_cnn.shape)
print("X_test_cnn shape :", X_test_cnn.shape)
print()
print("Response distribution:")
print(pd.Series(y).value_counts().sort_index())
print()
print("Example first 5 CRPs:")
print(df.head())
print()
print("Files saved:")
print("1. ro_puf_64bit_10000crps.npz")
print("2. ro_puf_64bit_10000crps.csv")