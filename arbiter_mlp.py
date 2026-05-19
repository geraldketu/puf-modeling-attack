

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import time


data = np.load("arbiter_puf_64bit_10000crps.npz")

y_train_np = data["y_train"]
y_val_np = data["y_val"]
y_test_np = data["y_test"]



X_raw_train = torch.tensor(data["X_raw_train"], dtype=torch.float32)
X_raw_val = torch.tensor(data["X_raw_val"], dtype=torch.float32)
X_raw_test = torch.tensor(data["X_raw_test"], dtype=torch.float32)



scaler = StandardScaler()

X_phi_train_np = scaler.fit_transform(data["X_phi_train"])
X_phi_val_np = scaler.transform(data["X_phi_val"])
X_phi_test_np = scaler.transform(data["X_phi_test"])

X_phi_train = torch.tensor(X_phi_train_np, dtype=torch.float32)
X_phi_val = torch.tensor(X_phi_val_np, dtype=torch.float32)
X_phi_test = torch.tensor(X_phi_test_np, dtype=torch.float32)



y_train = torch.tensor(y_train_np, dtype=torch.float32)
y_val = torch.tensor(y_val_np, dtype=torch.float32)
y_test = torch.tensor(y_test_np, dtype=torch.float32)

print("Loaded Arbiter PUF dataset")

print("X_raw_train:", X_raw_train.shape)
print("X_phi_train:", X_phi_train.shape)
print("y_train    :", y_train.shape)
print()



class PUF_MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dims=[128, 64, 32], dropout=0.2):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)



def train_model(
    X_tr,
    y_tr,
    X_v,
    y_v,
    X_te,
    y_te,
    name="MLP",
    epochs=100,
    lr=1e-3,
    batch_size=256
):
    model = PUF_MLP(input_dim=X_tr.shape[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True
    )

    best_val_acc = 0.0
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    print(f"Training: {name}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Val Acc':>10}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        # Training
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(X_tr)

        # Validation
        model.eval()

        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()

            val_probs = torch.sigmoid(val_logits)
            val_preds = (val_probs > 0.5).cpu().numpy().astype(int)

            val_acc = accuracy_score(y_v.cpu().numpy().astype(int), val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                key: value.clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>12.4f} {val_acc:>10.4f}")

        scheduler.step()

    
    model.load_state_dict(best_state)

    # Test
    model.eval()

    with torch.no_grad():
        test_logits = model(X_te)
        test_probs = torch.sigmoid(test_logits)
        test_preds = (test_probs > 0.5).cpu().numpy().astype(int)

        test_acc = accuracy_score(
            y_te.cpu().numpy().astype(int),
            test_preds
        )

    elapsed = time.time() - start_time

    print()
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy           : {test_acc:.4f}")
    print(f"Training time           : {elapsed:.2f} s")
    print()

    return model, history, best_val_acc, test_acc



# 7. MLP 

results = []

model_raw, hist_raw, val_raw, test_raw = train_model(
    X_raw_train,
    y_train,
    X_raw_val,
    y_val,
    X_raw_test,
    y_test,
    name="MLP on X_raw",
    epochs=100,
    lr=1e-3,
    batch_size=256
)

results.append(("MLP (X_raw)", val_raw, test_raw))


model_phi, hist_phi, val_phi, test_phi = train_model(
    X_phi_train,
    y_train,
    X_phi_val,
    y_val,
    X_phi_test,
    y_test,
    name="MLP on X_phi",
    epochs=100,
    lr=1e-3,
    batch_size=256
)

results.append(("MLP (X_phi)", val_phi, test_phi))





print("SUMMARY")

print(f"{'Model':<20} {'Best Val Acc':>14} {'Test Acc':>10}")
print("-" * 50)

for name, val, test in results:
    print(f"{name:<20} {val:>14.4f} {test:>10.4f}")