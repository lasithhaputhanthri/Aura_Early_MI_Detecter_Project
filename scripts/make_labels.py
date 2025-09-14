import pandas as pd

df = pd.read_csv("data/labels.csv")

# Remove 'Mask_' prefix and extra spaces
df["patient_id"] = df["patient_id"].str.replace("Mask_", "").str.replace(" ", "")

df.to_csv("data/labels_fixed.csv", index=False)
print("✅ Cleaned patient_id saved to labels_fixed.csv")

