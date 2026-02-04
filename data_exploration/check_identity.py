import pandas as pd

from constants import MIBI_GUT_DIR_PATH

base = MIBI_GUT_DIR_PATH
df = pd.read_csv(f"{base}/biopsy_metadata.csv")

# Filter rows where they differ
diff = df[df["Pathological stage"] != df["Grade GVHD"]]

if diff.empty:
    print("YES. They are identical in all 59 rows.")
else:
    print(f"NO. They differ in {len(diff)} rows.")
    print("\nSample Discrepancies:")
    print(diff[["Biopsy_ID", "Pathological stage", "Grade GVHD"]].head(10))

# Correlation check
print("\nCorrelation Matrix:")
print(df[["Pathological stage", "Grade GVHD"]].corr())
