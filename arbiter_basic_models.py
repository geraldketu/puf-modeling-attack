

import numpy as np
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report



data = np.load("arbiter_puf_64bit_10000crps.npz")

X_phi_train = data["X_phi_train"]
X_phi_val = data["X_phi_val"]
X_phi_test = data["X_phi_test"]

X_raw_train = data["X_raw_train"]
X_raw_val = data["X_raw_val"]
X_raw_test = data["X_raw_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

print("Loaded Arbiter PUF dataset")

print(f"Train: {X_phi_train.shape}")
print(f"Val  : {X_phi_val.shape}")
print(f"Test : {X_phi_test.shape}")
print()

results = []



print("MODEL 1: Logistic Regression on X_phi")


t0 = time.time()

lr_phi = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs"
)

lr_phi.fit(X_phi_train, y_train)

elapsed = time.time() - t0

val_pred = lr_phi.predict(X_phi_val)
test_pred = lr_phi.predict(X_phi_test)

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}")
print(f"Train time         : {elapsed:.2f} s")
print()

results.append(("Logistic Regression (X_phi)", val_acc, test_acc))



# Logistic Regression on X_raw

print("MODEL 2: Logistic Regression on X_raw")


t0 = time.time()

lr_raw = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs"
)

lr_raw.fit(X_raw_train, y_train)

elapsed = time.time() - t0

val_pred = lr_raw.predict(X_raw_val)
test_pred = lr_raw.predict(X_raw_test)

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}")
print(f"Train time         : {elapsed:.2f} s")
print()

results.append(("Logistic Regression (X_raw)", val_acc, test_acc))



# Linear SVM on X_phi

print("MODEL 3: Linear SVM on X_phi")


t0 = time.time()

svm_linear = LinearSVC(
    max_iter=5000,
    C=1.0
)

svm_linear.fit(X_phi_train, y_train)

elapsed = time.time() - t0

val_pred = svm_linear.predict(X_phi_val)
test_pred = svm_linear.predict(X_phi_test)

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}")
print(f"Train time         : {elapsed:.2f} s")
print()

results.append(("Linear SVM (X_phi)", val_acc, test_acc))



# RBF SVM on X_raw

print("MODEL 4: RBF SVM on X_raw")


t0 = time.time()

svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

svm_rbf.fit(X_raw_train, y_train)

elapsed = time.time() - t0

val_pred = svm_rbf.predict(X_raw_val)
test_pred = svm_rbf.predict(X_raw_test)

val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy      : {test_acc:.4f}")
print(f"Train time         : {elapsed:.2f} s")
print()

results.append(("RBF SVM (X_raw)", val_acc, test_acc))



print("Classification report for Logistic Regression on X_phi:")
print(classification_report(y_test, lr_phi.predict(X_phi_test), digits=4))



print()
print("SUMMARY")

print(f"{'Model':<35} {'Val Acc':>10} {'Test Acc':>10}")
print("-" * 60)

for name, val, test in results:
    print(f"{name:<35} {val:>10.4f} {test:>10.4f}")