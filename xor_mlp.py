# MLP (Multilayer Perceptron) with PyTorch
# Run AFTER xor_generate_crps.py
# Usage: python xor_mlp.py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# -----------------------------
data = np.load("xor_arbiter_puf_64bit_4xor_50000crps.npz")

X_train = torch.tensor(data["X_raw_train"], dtype=torch.float32)
X_val   = torch.tensor(data["X_raw_val"],   dtype=torch.float32)
X_test  = torch.tensor(data["X_raw_test"],  dtype=torch.float32)
y_train = torch.tensor(data["y_train"],     dtype=torch.float32)
y_val   = torch.tensor(data["y_val"],       dtype=torch.float32)
y_test  = torch.tensor(data["y_test"],      dtype=torch.float32)

scaler = StandardScaler()
X_phi_train_np = scaler.fit_transform(data["X_phi_train"])
X_phi_val_np   = scaler.transform(data["X_phi_val"])
X_phi_test_np  = scaler.transform(data["X_phi_test"])

X_phi_train = torch.tensor(X_phi_train_np, dtype=torch.float32)
X_phi_val   = torch.tensor(X_phi_val_np,   dtype=torch.float32)
X_phi_test  = torch.tensor(X_phi_test_np,  dtype=torch.float32)

INPUT_DIM = X_train.shape[1]  # 64
print(f"Input dim: {INPUT_DIM} | Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}\n")

# -----------------------------
# 2. Model definition
# 3. Hidden Layers 3 
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
# 3. Training loop
# -----------------------------
def train_model(X_tr, y_tr, X_v, y_v, X_te, name="MLP", epochs=50, lr=1e-3, batch_size=256):
    model = PUF_MLP(input_dim=X_tr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    print(f"Training: {name}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>10}")
    

    t0 = time.time()
    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        #Train
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb, yb
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(X_tr)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            logits_val = model(X_v)
            val_loss = criterion(logits_val, y_v).item()
            preds_val = (torch.sigmoid(logits_val) > 0.5).cpu().numpy()
            val_acc = accuracy_score(y_v.numpy(), preds_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} {val_acc:>10.4f}")

        scheduler.step()

    #  test accuracy with best weights 
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds_test = (torch.sigmoid(model(X_te)) > 0.5).cpu().numpy()
    test_acc = accuracy_score(y_test.numpy(), preds_test)

    elapsed = time.time() - t0
    print(f"\nBest Val Acc: {best_val_acc:.4f}  |  Test Acc: {test_acc:.4f}  |  Time: {elapsed:.1f}s")

    return model, history, best_val_acc, test_acc

# -----------------------------
# 4. Run Loop on different datasets
# -----------------------------
results = []

#  MLP on raw {-1,+1} features
model_raw, hist_raw, val_raw, test_raw = train_model(
    X_train, y_train, X_val, y_val, X_test,
    name="MLP on X_raw ({-1,+1})", epochs=100)
results.append(("MLP (X_raw)", val_raw, test_raw))

model_phi, hist_phi, val_phi, test_phi = train_model(
    X_phi_train, y_train, X_phi_val, y_val, X_phi_test,
    name="MLP on X_phi (arbiter features)", epochs=100)
results.append(("MLP (X_phi)", val_phi, test_phi))

# -----------------------------
# 5. Summary
# -----------------------------

print("SUMMARY")
print("--------------------------------")
print(f"{'Model':<20} {'Best Val Acc':>14} {'Test Acc':>10}")
for name, val, test in results:
    print(f"{name:<20} {val:>14.4f} {test:>10.4f}")



