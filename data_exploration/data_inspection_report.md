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

**Resolution Analysis & Normalization Strategy:**
To determine the correct pixel-to-micron conversion, we analyzed the coordinate ranges for each FOV size.
*   **Findings:**
    *   **400µm FOVs:** Coordinates range **0 - 1024**.
    *   **800µm FOVs:** Coordinates range **0 - 2048**.
*   **Conclusion:** The **resolution is consistent** (~0.39 µm/pixel), but the image dimensions vary.
    *   Normalization must be **group-specific**: Divide 400µm images by 1024, and 800µm images by 2048.

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

**4. Check for Non-Square FOVs & Resolution Consistency**
```bash
venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/MIBIGutCsv/cell_table.csv', usecols=['fov', 'centroid_x', 'centroid_y']); meta = pd.read_csv('data/MIBIGutCsv/fovs_metadata.csv', usecols=['FOV', 'Size [um]']); df = df.merge(meta, left_on='fov', right_on='FOV'); print(df.groupby('Size [um]')[['centroid_x', 'centroid_y']].agg(['min', 'max']))"
```

**5. Check for Biopsy Overlap (Colon + Duodenum in same Patient)**
```bash
venv\Scripts\python.exe -c "import pandas as pd; df = pd.read_csv('data/MIBIGutCsv/biopsy_metadata.csv'); mixed_patients = df.groupby('Patient_ID')['Localization'].nunique(); print(f'Patients with mixed biopsies: {mixed_patients[mixed_patients > 1]}')"
```

**6. Analyze Class Balance by Organ (Stratification Viability)**
```bash
venv\Scripts\python.exe data_exploration\check_data_bias.py
```

**Stratification Analysis Results:**
*   **Total Biopsies:** 73 (23 Colon, 50 Duodenum)
*   **Sample Composition:** 14 Controls, 59 GVHD patients
*   **Viable Strata:** 10/18 target×organ combinations (55.6%)

**Viable Targets for Correlation Analysis:**

| Target | Colon | Duodenum | Notes |
| :--- | :---: | :---: | :--- |
| **Pathological stage** | 23 ✗ | 50 ✓ | Colon: stage 4 has only 2 biopsies |
| **GI stage** | 23 ✗ | 50 ✓ | Colon: stage 1 has only 1 biopsy |
| **liver stage** | 23 ✗ | 50 ✗ | Too few advanced cases in gut biopsies |
| **skin stage** | 23 ✗ | 50 ✗ | Too few advanced cases in gut biopsies |
| **Grade GVHD** | 23 ✗ | 50 ✗ | Grade 2 has only 1 biopsy total |
| **Cortico Response** | 23 ✓ | 50 ✓ | 3 balanced classes (Control/Responder/Non-responder) |
| **Survival at follow-up** | 23 ✓ | 50 ✓ | 3 balanced classes |
| **Clinical score** | 23 ✓ | 50 ✓ | Simplified 3-class severity (Control/Mild/Severe) |
| **Pathological score** | 23 ✓ | 50 ✓ | Simplified 3-class severity |

**Key Findings:**
*   **Colon Limitation:** Only 23 biopsies cannot support fine-grained 5-class targets (stages 0-4).
*   **Duodenum Advantage:** 50 biopsies sufficient for stage-based analyses.
*   **Simplified Targets Robust:** 3-class targets (Clinical/Pathological score, Response, Survival) viable in both organs.
*   **Systemic GVHD:** liver/skin stage rarely severe in gut biopsies (primarily GI-specific disease).
*   **Action:** Correlation analysis must enforce organ-specific eligibility; pooling organs would confound biological differences.

---

## 4. Methodology & Mitigation Strategies

To address the findings from the data inspection, the following protocols have been established for the mining pipeline:

1.  **Coordinate Normalization (Implemented):**
    *   To ensure spatial consistency across FOVs with varying physical sizes (400µm vs 800µm) but identical pixel resolutions, we normalize all pixel coordinates to microns.
    *   **Logic:** $Coordinate_{\mu m} = Coordinate_{px} \times \frac{Size_{\mu m}}{Resolution_{max}}$. This allows for resolution-independent distance calculations (e.g., neighbors within 30µm).

2.  **Cohort Stratification:**
    *   Given the distinct biological profiles of the large and small intestine, analysis is strictly stratified by organ.
    *   **Strategy:** GVHD samples are split based on their `Localization` metadata (Colon vs. Duodenum), while Control samples are grouped by their specific `Cohort` identifier. This prevents confounding organ-specific signals with disease signals.
