import pandas as pd
import glob
import os

from constants import MIBI_GUT_DIR_PATH

try:
    path = glob.glob(MIBI_GUT_DIR_PATH + "biopsy_metadata.csv")[0]
    df = pd.read_csv(path)
    print("Columns:", df.columns.tolist())
    
    match = [c for c in df.columns if "pronoze" in c.lower() or "phase" in c.lower()]
    print("Matches:", match)
except Exception as e:
    print(e)
