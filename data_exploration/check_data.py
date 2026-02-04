import pandas as pd
import glob

from constants import MIBI_GUT_DIR_PATH

base = MIBI_GUT_DIR_PATH
biopsy = pd.read_csv(f"{base}/biopsy_metadata.csv")
fovs = pd.read_csv(f"{base}/fovs_metadata.csv")

print("--- Columns ---")
print(biopsy.columns.tolist())

print("\n--- Unique Values: Days after Transplant grouped ---")
print(biopsy["Days after Transplant grouped"].unique())

print("\n--- Unique Values: Pathological stage ---")
print(biopsy["Pathological stage"].unique())

print("\n--- Join Check ---")
fov_patients = set(fovs["Patient"].unique())
bio_ids = set(biopsy["Biopsy_ID"].unique())
pat_ids = set(biopsy["Patient_ID"].unique())

print(f"FOV Patients: {len(fov_patients)} samples (e.g. {list(fov_patients)[:3]})")
print(f"Biopsy IDs: {len(bio_ids)} samples (e.g. {list(bio_ids)[:3]})")
print(f"Patient IDs: {len(pat_ids)} samples (e.g. {list(pat_ids)[:3]})")

print(f"Intersection FOV-Patient <-> Biopsy-BiopsyID: {len(fov_patients.intersection(bio_ids))}")
print(f"Intersection FOV-Patient <-> Biopsy-PatientID: {len(fov_patients.intersection(pat_ids))}")
