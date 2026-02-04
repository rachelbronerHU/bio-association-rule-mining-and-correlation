import pandas as pd
import glob

from constants import MIBI_GUT_DIR_PATH

base = MIBI_GUT_DIR_PATH
df = pd.read_csv(f"{base}/biopsy_metadata.csv")

candidates = [
    "Pathological stage",
    "Grade GVHD",
    "Clinical score",
    "Pathological score",
    "Cortico Response",
    "Survival at follow-up",
    "GI stage",
    "Conditioning", 
    "Diagnosis"
]

print(f"Total Rows: {len(df)}")
print("-" * 30)

for col in candidates:
    if col in df.columns:
        uniques = df[col].unique()
        # Sort if possible for cleaner display
        try:
            uniques = sorted([x for x in uniques if str(x) != 'nan'])
        except:
            pass
        print(f"COLUMN: '{col}'")
        print(f"Unique Count: {len(uniques)}")
        print(f"Values: {uniques}")
        print("-" * 30)
