import os
import glob
import numpy as np
import pandas as pd
import scipy.io as sio

INPUT_DIR = "data/raw/"
OUTPUT_DIR = "data/processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

labels = []

def extract_area(mask):
    return np.sum(mask)

for path in sorted(glob.glob(INPUT_DIR + "*.mat")):
    mat = sio.loadmat(path)
    pid = os.path.basename(path).split(".")[0].strip()  # keep full, e.g. ES0001_4CH_1

    # auto detect key:
    possible_keys = [k for k in mat.keys() if not k.startswith("__")]
    array_key = None
    for k in possible_keys:
        if isinstance(mat[k], np.ndarray) and mat[k].ndim >= 2:
            array_key = k
            break
    if not array_key:
        print(f"❌ {pid} no array")
        continue

    frames = mat[array_key]
    areas = [extract_area(f) for f in frames]
    areas = np.array(areas)

    EDV = np.max(areas)
    ESV = np.min(areas)
    LVEF = (EDV - ESV) / EDV * 100 if EDV > 0 else 0

    seq = areas.reshape(-1, 1)
    if len(seq) < 32:
        seq = np.pad(seq, ((0, 32 - len(seq)), (0, 0)), mode="edge")
    elif len(seq) > 32:
        seq = seq[:32]

    np.save(os.path.join(OUTPUT_DIR, f"{pid}.npy"), seq)
    print(f"✅ {pid} | LVEF={LVEF:.2f}% | shape={seq.shape}")

    label = int(LVEF < 20)
    labels.append([pid, round(LVEF, 2), label])

df = pd.DataFrame(labels, columns=["patient_id", "LVEF", "label"])
df.to_csv("data/labels.csv", index=False)
print("🎉 DONE: labels.csv matches .npy filenames.")

