#ML Baselines (Logistic Regression + SVM)
# Run AFTER xor_generate_crps.py
# Usage: python3 xor_basic_models

#Dependencies:
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import time

# -----------------------------
# 1. Load data (USE DATA FROM xor_generate_crps.py)
# -----------------------------
data = np.load("xor_arbiter_puf_64bit_4xor_50000crps.npz")

X_phi_train = data["X_phi_train"]
X_phi_val   = data["X_phi_val"]
X_phi_test  = data["X_phi_test"]

X_raw_train = data["X_raw_train"]
X_raw_val   = data["X_raw_val"]
X_raw_test  = data["X_raw_test"]

y_train = data["y_train"]
y_val   = data["y_val"]
y_test  = data["y_test"]

print(f"Train: {X_phi_train.shape} | Val: {X_phi_val.shape} | Test: {X_phi_test.shape}\n")

results = []


# 2. Logistic Regression on X_phi
# Create a model train on x_phi crps.
# x_phi is the transformation that can sorta emulate the structure of a PUF 
# -----------------------------

print("MODEL 1: Logistic Regression on X_phi ")
print("--------------------------------------")
t0 = time.time()
lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs") #1000 iterations at lest , solver is algorithm for finding best weights
lr.fit(X_phi_train, y_train)
elapsed = time.time() - t0  #keep track of time 

val_acc  = accuracy_score(y_val,  lr.predict(X_phi_val))
test_acc = accuracy_score(y_test, lr.predict(X_phi_test))
print(f"Val  Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  |  Train time: {elapsed:.1f}s")
#print(classification_report(y_test, lr.predict(X_phi_test), digits=4))
results.append(("LR (X_phi)", val_acc, test_acc))


# 3. Logistic Regression on X_raw
#  raw {-1,+1} 
# -----------------------------

print("MODEL 2: Logistic Regression on X_raw (raw bits)")
print("------------------------------------------------")
t0 = time.time()
lr_raw = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
lr_raw.fit(X_raw_train, y_train)
elapsed = time.time() - t0

val_acc  = accuracy_score(y_val,  lr_raw.predict(X_raw_val))
test_acc = accuracy_score(y_test, lr_raw.predict(X_raw_test))
print(f"Val  Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  |  Train time: {elapsed:.1f}s")
results.append(("LR (X_raw)", val_acc, test_acc))


# 4. Linear SVM on X_phi
# -----------------------------
print("MODEL 3: Linear SVM on X_phi")
print("----------------------------")
t0 = time.time()
svm_lin = LinearSVC(max_iter=5000, C=1.0)
svm_lin.fit(X_phi_train, y_train)
elapsed = time.time() - t0

val_acc  = accuracy_score(y_val,  svm_lin.predict(X_phi_val))
test_acc = accuracy_score(y_test, svm_lin.predict(X_phi_test))
print(f"Val  Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  |  Train time: {elapsed:.1f}s")
results.append(("LinearSVM (X_phi)", val_acc, test_acc))

# 5. RBF SVM on X_raw (subset — RBF is slow on 50k)
# -----------------------------

print("MODEL 4: RBF SVM on X_raw")
print("--------------------------")
subset = 10000
t0 = time.time()
svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale")
svm_rbf.fit(X_raw_train[:subset], y_train[:subset])
elapsed = time.time() - t0

val_acc  = accuracy_score(y_val,  svm_rbf.predict(X_raw_val))
test_acc = accuracy_score(y_test, svm_rbf.predict(X_raw_test))
print(f"Val  Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}  |  Train time: {elapsed:.1f}s  (trained on {subset} samples)")
results.append(("RBF SVM (X_raw, 10k)", val_acc, test_acc))

# -----------------------------
# 6. Summary 
print("\n" + "------------------------")
print("SUMMARY")
print("------------------------")
print(f"{'Model':<25} {'Validation Accuracy':>10} {'Test Accuracy':>10}")
print("-" * 45)
for name, val, test in results:
    print(f"{name:<25} {val:>10.4f} {test:>10.4f}")
