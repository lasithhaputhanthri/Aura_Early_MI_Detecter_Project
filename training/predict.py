import os
import numpy as np
import torch
from models.model import CNN_BiLSTM_Classifier
import json

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pt"

# Load saved parameters
with open("models/best_params.json", "r") as f:
    best_params = json.load(f)

# Create model with correct params
model = CNN_BiLSTM_Classifier(
    dropout=best_params["dropout"],
    hidden_size=best_params["hidden_size"]
)
model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
model.eval()

# Choose a file to test
test_file = "data/processed/ES0001_4CH_1.npy"
x = np.load(test_file)

# Pad/clip to 32 frames
if x.shape[0] < 32:
    pad = np.repeat(x[-1:], 32 - x.shape[0], axis=0)
    x = np.concatenate([x, pad], axis=0)
elif x.shape[0] > 32:
    x = x[:32]

# Predict
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    logits = model(x_tensor).squeeze()
    prob = torch.sigmoid(logits).item()
    pred = int(prob >= 0.5)

# Output
print(f"✅ Predicted MI Probability: {prob:.3f}")
print(f"🧠 Predicted Class: {'MI' if pred == 1 else 'No MI'}")