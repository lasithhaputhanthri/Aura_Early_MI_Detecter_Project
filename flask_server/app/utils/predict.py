import sys
import time
import random
import os
import numpy as np
import torch
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.model import CNN_BiLSTM_Classifier

# def predictMI(filepath):
#     print(f"[INFO] Simulating processing for: {filepath}")
#     time.sleep(2)  # Simulate some work
#     return {
#         "MI": random.choice([True, False]),
#         "confidence": round(random.uniform(0.4, 0.9), 2)
#     }

def predictMI():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")
    PARAM_PATH = os.path.join(BASE_DIR, "models", "best_params.json")

    # Load saved parameters
    with open(PARAM_PATH, "r") as f:
        best_params = json.load(f)

    # Create model
    model = CNN_BiLSTM_Classifier(
        dropout=best_params["dropout"],
        hidden_size=best_params["hidden_size"]
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Input path
    default_input = os.path.join(BASE_DIR, "flask_server","ActivePolyOut", "scaled_areas.npy")
    print(default_input)
    file_to_use =default_input

    # Load data
    # Load the data
    x = np.load(file_to_use, allow_pickle=True)

    # Determine how many frames are available
    num_frames = x.shape[0]

    if num_frames < 32:
        # Pad with the last frame to reach 32
        pad = np.repeat(x[-1:], 32 - num_frames, axis=0)
        x = np.concatenate([x, pad], axis=0)
    else:
        # Uniformly sample 32 frames
        indices = np.linspace(0, num_frames - 1, 32).astype(int)
        x = x[indices]


    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(x_tensor).squeeze()
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= 0.5)

    print(f"✅ Predicted MI Probability: {prob:.3f}")
    print(f"🧠 Predicted Class: {'MI' if pred == 1 else 'No MI'}")

    return {
        "MI": bool(pred),
        "confidence": prob
    }
