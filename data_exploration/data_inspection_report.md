# Data Inspection & Biological Context Report

**Date:** 15 February 2026
**Project:** Script Rule Mining (GVHD Analysis)

## 1. Executive Summary

This report documents the structure and biological meaning of the provided dataset, highlighting critical issues that must be addressed before further mining. 

**Key Findings:**
1.  **Coordinate Scaling Issue:** FOVs have different physical sizes (400µm vs 800µm) but share the same pixel resolution (2048x2048). **Action Required:** Normalize coordinates to microns.
2.  **Mixed Cohorts:** The disease group (`Cohort_GVHD`) contains samples from both the **Colon** and **Duodenum**, while controls are separated. **Action Required:** Split analysis by organ.
3.  **Spatial Context:** The dataset contains explicit anatomical masks (`in_CryptVilli`, `in_Muscle`, etc.) **Action Required:** Integrate these regions into transaction generation.

---

## 2. Metadata Analysis

### A. FOV Metadata (`fovs_metadata.csv`)

| Column | Unique Values | Meaning & Action |
| :--- | :--- | :--- |
| **Cohort** | `Duodenum_Cohort_Control_1`, `Duodenum_Cohort_Control_2`, `Colon_Cohort_Control`, `Cohort_GVHD` | Defines the experimental groups. **Issue:** `Cohort_GVHD` is mixed. **Fix:** Split GVHD by `Localization` in Biopsy Metadata. |
| **Size [um]** | `400`, `800` | Physical side length of the image. **Critical:** Pixel coordinates (0-2048) must be converted to microns using this value. |

### B. Biopsy Metadata (`biopsy_metadata.csv`)

This file contains clinical covariates crucial for rule mining.

#### **Disease & Diagnosis**
*   **`Diagnosis`** (`AML`, `MDS`): Underlying cancer pre-transplant.
*   **`Localization`** (`Colon`, `Duodenum`): **The primary splitter.** Determines if the biopsy is from the small or large intestine.
*   **`Grade GVHD`** (`0-3`) / **`Pathological stage`** (`0-3`): Clinical and histological severity scores. Useful as target variables.
*   **`Clinical score` / `Pathological score`** (`Severe`, `Mild`): Simplified binary targets.

#### **Treatment & History**
*   **`Conditioning`**: (`RIC`, `Myeloablative`) Intensity of pre-transplant chemotherapy.
*   **`GVHD Prophylaxis`**: (`CSA MMF`, `CSA MTX`) Drugs used to prevent GVHD.
*   **`Immunosuppressants...`**: (`Yes`, `No`) Patient status at time of biopsy.

#### **Outcomes**
*   **`Cortico Response`** (`Responder`, `Non-responder`): **High-value target.** Did the patient respond to steroid treatment?
*   **`Survival at follow-up`** (`Survived`, `Not survived`): Hard endpoint for prognosis rules.
*   **`Virus in biopsy`** (`Negative`, `CMV`, etc.): Potential confounder for inflammation.

### C. Cell Table (`cell_table.csv`)

*   **`cell type` / `population`**: Cell identity (e.g., T-Cell, B-Cell).
*   **Anatomical Masks (`in_*`)**: 
    *   `in_CryptVilli` (Epithelium)
    *   `in_Lumen` (Gut interior)
    *   `in_Muscle` (Muscularis mucosae)
    *   `in_SMV` (Submucosal Vein)
    *   `in_LP` (Lamina Propria)
    *   `in_Follicle` (Lymphoid follicle)
    
    **Strategy:** Use these boolean columns to spatially tag cells (e.g., "CD8 T-cell in Epithelium") instead of relying on geometric slicing direction.

---

## 3. Data Exploration Commands

The following Python commands were executed to extract the unique values and inspect the data structure.

**1. Inspect FOV Metadata (Cohorts & Sizes)**
```bash
venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/MIBIGutCsv/fovs_metadata.csv'); [print(f'{col}: {df[col].unique()}') for col in df.columns if df[col].nunique() < 50]"
```

**2. Inspect Biopsy Metadata (Clinical Variables)**
```bash
venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/MIBIGutCsv/biopsy_metadata.csv'); [print(f'{col}: {df[col].unique()}') for col in df.columns if df[col].nunique() < 50]"
```

**3. Inspect Cell Table Columns**
```bash
venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/MIBIGutCsv/cell_table.csv', nrows=5); print(f'Columns: {list(df.columns)}')"
```

---

## 4. Recommended Next Steps

1.  **Coordinate Normalization:** Update `load_data` to convert `centroid_x` and `centroid_y` to microns: `coord_micron = coord_pixel * (Size_um / max(coord_pixel))` (dynamically determined per FOV).
2.  **Cohort Splitting:** Implement a filter to process `Colon` and `Duodenum` samples separately.
3.  **Region Integration:** Update `transactions.py` to include anatomical regions (`in_CryptVilli`, etc.) as items in the transaction sets.
