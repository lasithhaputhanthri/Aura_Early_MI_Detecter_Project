import os
import pandas as pd

# === Load valid patient IDs ===
df = pd.read_csv("data/labels_fixed.csv")
good_ids = set(df["patient_id"].astype(str).str.strip())
print(f"✅ Loaded {len(good_ids)} valid IDs.")

# === Paths ===
INPUT_DIR = "data/processed"

renamed = 0
deleted = 0
skipped = 0

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".npy"):
        continue

    old_path = os.path.join(INPUT_DIR, filename)
    # Clean the name: remove Mask_, spaces, .npy
    base = filename.replace(".npy", "").replace("Mask_", "").replace(" ", "")

    # Try to find a match
    matches = [pid for pid in good_ids if pid == base or base in pid or pid in base]

    if matches:
        # If multiple, take the first (could improve with better match logic)
        new_name = matches[0] + ".npy"
        new_path = os.path.join(INPUT_DIR, new_name)

        if old_path == new_path:
            print(f"✅ Already correct: {filename}")
            continue

        if os.path.exists(new_path):
            print(f"⚠️ Skipping rename: {new_name} already exists.")
            skipped += 1
            continue

        os.rename(old_path, new_path)
        print(f"✅ Renamed {filename} -> {new_name}")
        renamed += 1
    else:
        # No match → safe to delete
        os.remove(old_path)
        print(f"🗑️ Deleted unmatched file: {filename}")
        deleted += 1

print(f"\n🎉 Done! Renamed: {renamed} | Deleted: {deleted} | Skipped: {skipped}")

