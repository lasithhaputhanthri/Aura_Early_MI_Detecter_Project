import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import optuna
from models.model import CNN_BiLSTM_Classifier
import json


# Config
SEED = 42
N_SPLITS = 5
EPOCHS = 30
PATIENCE = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load Data
X_list, Y_list = [], []
labels = {}
with open("data/labels_fixed.csv") as f:
    next(f)
    for line in f:
        pid, _, label = line.strip().split(",")
        labels[pid] = int(label)

for file in os.listdir("data/processed"):
    if file.endswith(".npy"):
        pid = file.replace(".npy", "").replace("Mask_", "")
        if pid in labels:
            X_list.append(np.load(f"data/processed/{file}"))
            Y_list.append(labels[pid])

X = torch.tensor(np.stack(X_list), dtype=torch.float32)
Y = torch.tensor(Y_list, dtype=torch.float32).unsqueeze(1)

# Create model factory
def create_model(dropout, hidden_size):
    return CNN_BiLSTM_Classifier(dropout=dropout, hidden_size=hidden_size)

# Focal Loss
def focal_loss(alpha=1, gamma=2):
    class FocalLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.bce = nn.BCEWithLogitsLoss(reduction='none')
        def forward(self, inputs, targets):
            bce = self.bce(inputs, targets)
            pt = torch.exp(-bce)
            return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
    return FocalLoss()

# Globals for saving the best model
BEST_GLOBAL_ACC = 0
BEST_MODEL_STATE = None
BEST_PARAMS = None

# Optuna objective
def objective(trial):
    global BEST_GLOBAL_ACC, BEST_MODEL_STATE, BEST_PARAMS

    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, Y.squeeze().numpy())):
        model = create_model(dropout, hidden_size).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = focal_loss()

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        class_counts = torch.tensor([(Y_train == t).sum() for t in torch.unique(Y_train)])
        weights = 1.0 / class_counts.float()
        sample_weights = torch.tensor([weights[int(t)] for t in Y_train])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

        best_acc = 0
        early_stop = 0

        for epoch in range(EPOCHS):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = model(bx).squeeze(1)
                loss = criterion(out, by.squeeze())
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(DEVICE)
                    logits = model(bx).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds.extend(probs)
                    val_labels.extend(by.numpy())

            preds = (np.array(val_preds) >= 0.5).astype(int)
            acc = accuracy_score(val_labels, preds)
            if acc > best_acc:
                best_acc = acc
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= PATIENCE:
                    break

        val_accuracies.append(best_acc)

    mean_acc = np.mean(val_accuracies)

    # Save only the best model across all trials
    if mean_acc > BEST_GLOBAL_ACC:
        BEST_GLOBAL_ACC = mean_acc
        BEST_MODEL_STATE = model.state_dict()
        BEST_PARAMS = {
            'dropout': dropout,
            'hidden_size': hidden_size,
            'lr': lr,
            'batch_size': batch_size
        }

    return mean_acc

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

# Save best model
torch.save(BEST_MODEL_STATE, "models/best_model.pt")

# Save best params
with open("models/best_params.json", "w") as f:
    json.dump(BEST_PARAMS, f)

print("\n✅ Best Hyperparameters:")
print(BEST_PARAMS)
print(f"📊 Mean Val Accuracy: {BEST_GLOBAL_ACC:.4f}")
print("📁 Saved as best_model.pt and best_params.json")

