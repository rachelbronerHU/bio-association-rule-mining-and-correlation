# Stratification & Analysis Refactor Plan

**Objective:** Refactor `check_rule_correlation_with_disease` to compare only biologically comparable groups.

Separated into **DONE** (✅ complete), **TODO** (needs implementation), and **NICE TO HAVE** (optional improvements).

---

## ✅ DONE

### 0) Data bias check — per organ
**Status:** ✅ Complete (`data_exploration/check_data_bias.py`)

**What it does:**
* Analyzes class balance for each `target × organ` combination
* Flags non-viable strata where:
  * Any class has <3 biopsies, OR
  * Majority class >80% of stratum
* Saves `bias_summary_by_organ.csv` with viability flags and reasons
* Current results: **10/18 strata viable** (55.6%)

**Viability criteria:**
* Min 3 biopsies per class (for LOGO CV)
* Max 80% majority class (prevents severe imbalance)
* Note: Does NOT check class ratios (some viable strata have 5-7:1 ratios) - this is acceptable with proper imbalance handling (see Task 4)

### 8) Consensus flow stratification
**Status:** ✅ Complete (`result_exploration/generate_consensus_tables.py`, `visualization/visualize_consensus.py`)

**What it does:**
* Uses `load_stratified_biopsies()` from `check_data_bias.py` for organ data
* Auto-discovers and loops through all available organs
* Generates separate consensus tables/plots per organ (`_Colon.csv`, `_Duodenum.csv`)
* Includes bias flags as metadata (warnings in plots, not gates)
* Exploration mode: generates all outputs regardless of viability

---

## 📋 TODO (required for valid comparisons)

### 1) Use existing stratification infrastructure ✅
**Status:** ✅ Complete
**What to use:** `load_stratified_biopsies()` from `data_exploration/check_data_bias.py`

This function already provides:
* `Biopsy_ID`, `Organ` (Colon/Duodenum), `Is_Control` flag
* All target columns with proper NaN handling
* Derived from biopsy metadata + FOV data

**No need to create new `stratified_utils.py` - reuse existing code.**

### 2) Refactor correlation scripts with organ stratification ✅
**Status:** ✅ Complete (`run_robust_simple_stats.py`, `advanced_discovery.py`)
**Files to modify:** `check_rule_correlation_with_disease/run_robust_simple_stats.py`, `advanced_discovery.py`

**Changes needed:**
1. Import stratification: `from data_exploration.check_data_bias import load_stratified_biopsies`
2. Load results CSV and merge with stratification data on `Biopsy_ID`
3. Add organ loop: `for organ in ['Colon', 'Duodenum']`
4. Apply target-specific eligibility (see table below)
5. Save outputs with organ suffix (e.g., `_Colon_Pathological_stage.csv`)

**Target eligibility rules:**
Controls are valid for severity targets (control = stage 0) but NOT for treatment/outcome targets.

| Target | Include Controls? | Rationale |
|--------|-------------------|-----------|
| Pathological stage | ✅ Yes | Full severity spectrum 0→3 |
| GI stage | ✅ Yes | Full severity spectrum 0→3 |
| Grade GVHD | ✅ Yes | Full severity spectrum 0→3 |
| Clinical score | ✅ Yes | 3-class: Control→Mild→Severe |
| Pathological score | ✅ Yes | 3-class: Control→Mild→Severe |
| Cortico Response | ❌ No (GVHD only) | Controls not treated |
| Survival at follow-up | ❌ No (GVHD only) | Controls not transplant patients |

### 3) Add bias gating to correlation pipeline ✅
**Status:** ✅ Complete (via `is_viable_stratum()` in both scripts)
**What:** Load `bias_summary_by_organ.csv` and skip non-viable strata

**Implementation:**
```python
# At script start
bias_df = pd.read_csv("results/full_run/plots/data_bias_report/bias_summary_by_organ.csv")

# Before each target×organ analysis
def is_viable(target, organ, bias_df):
    row = bias_df[(bias_df['Target'] == target) & (bias_df['Organ'] == organ)]
    if row.empty:
        return False, "No bias data"
    return row.iloc[0]['Viable'] == '✓', row.iloc[0]['Reason']

# In analysis loop
viable, reason = is_viable(target, organ, bias_df)
if not viable:
    logger.info(f"⊘ Skipping {target} × {organ}: {reason}")
    continue
```

**Why:** Prevents running meaningless analyses on strata with <3 samples per class or >80% majority.

### 4) Enhance imbalance handling ✅
**Status:** ✅ Complete (`run_robust_simple_stats.py`, `advanced_discovery.py`)
**Current status:**
* ✅ `class_weight='balanced'` in RandomForest and LogisticRegression
* ✅ Macro-F1 score (treats all classes equally)
* ✅ XGBoost sample weights via `get_xgb_sample_weights`
* ✅ Per-class F1 via `compute_per_class_metrics` → `Per_Class_F1` column in leaderboard
* ✅ Bootstrap CIs via `_bootstrap_single_iteration` + batched `ProcessPoolExecutor` → `CI_Mean`, `CI_Lower`, `CI_Upper` columns

**Changes needed:**

**A) Add XGBoost class weighting:**
```python
# Calculate scale_pos_weight for binary, sample_weight for multiclass
if is_binary:
    scale = len(y[y==0]) / len(y[y==1])
    xgb.XGBClassifier(..., scale_pos_weight=scale)
else:
    # Compute sample weights manually
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight('balanced', y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
```

**B) Add per-class F1 scores in output:**
```python
from sklearn.metrics import classification_report
# After prediction
report = classification_report(y_true, y_pred, output_dict=True)
# Save per-class precision/recall/f1 to results CSV
```
*Why:* Shows if minority classes are actually learned (not just overall score)

**C) Add bootstrap confidence intervals:**
```python
from sklearn.utils import resample

def bootstrap_ci(X, y, groups, model, n_bootstrap=1000):
    """Resample biopsies (with replacement) and compute CI"""
    scores = []
    for _ in range(n_bootstrap):
        boot_groups = resample(np.unique(groups), replace=True)
        boot_idx = np.isin(groups, boot_groups)
        # Run LOGO CV on bootstrap sample
        score = evaluate_logo_cv(X[boot_idx], y[boot_idx], groups[boot_idx], model)
        scores.append(score)
    return np.mean(scores), np.percentile(scores, [2.5, 97.5])

# Usage
mean_f1, (ci_lower, ci_upper) = bootstrap_ci(X, y, groups, model)
# Report: F1 = 0.68 [0.64-0.72]
```
*Why:* Quantifies uncertainty due to small N (23-50 biopsies). Wide CI = less reliable result.

**Macro-F1 explained:** Calculates F1 per class, then averages them equally. Prevents ignoring minority classes.
**Bootstrap CI explained:** Resamples data 1000x to estimate variability. Narrow CI = stable, wide CI = uncertain.

### 5) Add metadata to outputs
**What to include in result files:**
* Organ stratum (Colon/Duodenum)
* Population (GVHD only / GVHD + Controls)
* Class counts (per target class)
* Viability flag from bias report
* Per-class F1 scores
* Bootstrap confidence intervals [lower, upper]

**Format:** Add columns to results CSV or save separate metadata JSON per analysis.

### 6) Implement dry-run mode ✅
**Status:** ✅ Complete (`check_rule_correlation_with_disease/run_correlation_pipeline.py --dry_run`)
**Add flag:** `--dry_run` to both correlation scripts

**What it prints:**
```
Dry Run Mode - No modeling will be performed
================================================
Organ: Colon (23 biopsies, 74 FOVs)
  GVHD: 19, Controls: 4

Target: Pathological stage
  ✓ VIABLE - Will run
  Population: GVHD + Controls (eligibility: Yes)
  Classes: {0:4, 1:8, 2:5, 3:4, 4:2}
  
Target: Grade GVHD
  ✗ SKIP - Class 1.0 has only 1 biopsies (need ≥3)
  
Organ: Duodenum (50 biopsies, 156 FOVs)
  ...

Summary: 10 analyses will run, 8 will be skipped
```

**Purpose:** Verify stratification logic before expensive modeling runs.

---

## 🎯 NICE TO HAVE (optional improvements)

### A) Biopsy-level feature aggregation
Current: FOV-level features with biopsy-grouped CV (overweights biopsies with many FOVs)
Improvement: Aggregate FOV features to one row per biopsy (mean/max/presence) before modeling

### B) Confounder adjustment
Add covariates (diagnosis, conditioning, prophylaxis) to models within each stratum to reduce bias

### C) Within-stratum permutation null baselines
Shuffle target labels within organ, rerun evaluation, compare observed vs null distribution

### D) Stability-first rule ranking
Rank rules by cross-fold selection frequency, not only single-run importance

---

## 📝 Summary of Changes

**What's done:** ✅
- Bias checking per organ with viability flags
- Consensus flow fully stratified by organ
- Class weights in RF/Lasso, Macro-F1 scoring

**What's done (Tasks 1–4, 6):** ✅
- Correlation scripts refactored with organ stratification and bias gating
- XGB sample weights, per-class F1, bootstrap CIs in leaderboard output
- Dry-run mode via `run_correlation_pipeline.py --dry_run`

**Key decisions:**
- Reuse `load_stratified_biopsies()` from `check_data_bias.py` (no new files needed)
- Keep viability at min=3, max_majority=80% (no ratio check needed with proper imbalance handling)
- Bootstrap CIs quantify uncertainty for small-N data (23-50 biopsies per organ)
