import pandas as pd

from constants import MIBI_GUT_DIR_PATH

base = MIBI_GUT_DIR_PATH
df = pd.read_csv(f"{base}/biopsy_metadata.csv")

# Focus on rows where Clinical (Grade GVHD) != Pathological
mismatch = df[df["Pathological stage"] != df["Grade GVHD"]].copy()

# Select columns that might explain the difference
cols = [
    "Biopsy_ID", 
    "Localization",          # Where was this specific biopsy taken?
    "Grade GVHD",            # Global Clinical Score
    "Pathological stage",    # Score of THIS biopsy
    "GI stage",              # Gut Score
    "liver stage", 
    "skin stage"
]

print(f"Analyzing {len(mismatch)} mismatches...")
print(mismatch[cols].head(15).to_string(index=False))
