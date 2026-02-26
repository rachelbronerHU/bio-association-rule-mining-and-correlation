# Stratification & Analysis Refactor Plan

**Objective:** Refactor `check_rule_correlation_with_disease` to compare only biologically comparable groups.

Separated into **MUST** (without these, results are invalid) and **NICE TO HAVE** (stronger validity).

---

## MUST (required for valid comparisons)

### 0) Data bias check — per organ (run first, before any correlation analysis)

**Step 0a — Enhance `data_exploration/check_data_bias.py`:**
* Current script checks class balance globally (all biopsies pooled).
* Problem: a target can look balanced globally but be severely skewed within one organ (e.g., Colon has 80% stage 0, Duodenum has 80% stage 3).
* Changes:
  * Add `Localization` (organ) as a breakdown dimension.
  * For each target, show class distribution **per organ**, not just global.
  * Flag any `target × organ` combination where:
    * Any class has **<3 biopsies**, OR
    * Majority class is **>80%** of the stratum.
  * Save `bias_summary_by_organ.csv` with per-organ imbalance scores and flags.

**Step 0b — Use bias report as a gate in the correlation pipeline:**
* Before running any `target × organ` analysis, load `bias_summary_by_organ.csv`.
* Skip (and log with reason) any `target × organ` combination flagged as too biased or too small.
* This prevents the pipeline from producing meaningless results on severely imbalanced strata.

### 1) Enforce strict organ stratification
* Compare Colon only with Colon, Duodenum only with Duodenum.
* Never run a pooled Colon+Duodenum comparison.

### 2) Define target-specific eligibility (who is included per target)

Controls are valid for severity-scale targets (where control = stage 0 / grade 0), but **not** for treatment-response or outcome targets (controls were never treated / aren't transplant patients).

| Target | Include Controls? | Population | Rationale |
| :--- | :--- | :--- | :--- |
| `Pathological stage` | ✅ Yes (as stage 0) | GVHD + Controls | Full severity spectrum 0→3 |
| `GI stage` | ✅ Yes (as stage 0) | GVHD + Controls | Same logic |
| `Grade GVHD` | ✅ Yes (as grade 0) | GVHD + Controls | Same logic |
| `Cortico Response` | ❌ No | GVHD only | Controls were never treated with steroids |
| `Survival at follow-up` | ❌ No | GVHD only | Controls aren't transplant patients |
| `Clinical score` | ✅ Yes (as "Control") | GVHD + Controls | 3-class: Control → Mild → Severe |
| `Pathological score` | ✅ Yes (as "Control") | GVHD + Controls | 3-class: Control → Mild → Severe |

### 3) Build one shared stratification loader
* **File:** `check_rule_correlation_with_disease/stratified_utils.py` (New)
* **Function:** `load_stratified_data(results_file_path)`
* **Core logic:**
  1. Load results CSV (already contains `Cohort`, `Biopsy_ID`, `FOV`, targets).
  2. Add `Localization` only — merge from `biopsy_metadata.csv` on `Biopsy_ID`. For controls (no biopsy metadata), derive organ from `Cohort` name directly.
  3. Create `Organ` column (`Colon` or `Duodenum`).
  4. Create `Is_Control` column from `Cohort`.
  5. Assert no unknown/unmapped rows — fail loudly, never silently assign.

### 4) Refactor both analysis scripts to use the same strata
* **Files:** `run_robust_simple_stats.py`, `advanced_discovery.py`
* **Execution pattern:**
  * Outer loop: `for organ in [Colon, Duodenum]`
  * Inner loop: `for target in TARGETS`
  * Before each analysis: apply eligibility table (point 2) to include/exclude controls.
  * Both scripts must call `load_stratified_data()` so they operate on identical rows.
* Save outputs with organ suffix (e.g., `_Colon_Pathological_stage`).

### 5) Minimum sample guard
* After filtering by organ + target + eligibility, check class counts.
* Require **≥3 biopsies per class** to run LOGO CV. Otherwise skip and log reason.
* This prevents meaningless evaluation on tiny strata.

### 6) Comparability metadata in outputs
* Each result file must include:
  * organ stratum
  * included groups (GVHD only / GVHD + Controls)
  * biopsy count per class
  * FOV count per class
* If a comparison was skipped, log it with reason.

### 7) Verification before full run
* Implement `--dry_run` flag (does not exist yet).
* Dry run prints:
  * row counts per `Organ × Is_Control`
  * class counts per `Organ × Target`
  * which analyses will run vs skip (with reason)
* No modeling, no file output.

### 8) Consensus flow alignment (exploration-first, stratified outputs)
* **Scope:** `result_exploration/generate_consensus_tables.py` + `visualization/visualize_consensus.py`
* Add a dedicated stratified consensus mode built on the same `load_stratified_data()` output so consensus and correlation analyses are comparable.
* Default behavior must auto-discover strata from data (`sorted(df["Organ"].dropna().unique())`) and loop all organs; do not require organ as a mandatory CLI param.
* Keep optional organ filtering only as a convenience (e.g., `--organs Colon Duodenum`) for focused reruns.
* For consensus/exploration, do **not** apply hard exclusion gates (no skip due to imbalance/small counts); generate outputs for every available organ.
* Instead of gating, attach reliability context everywhere:
  * include biopsy/FOV counts per stratum in consensus tables
  * carry imbalance/low-count flags from `bias_summary_by_organ.csv`
  * show warning text in plot titles/subtitles/footnotes when strata are weak
* Save outputs with explicit organ suffixes so files are not pooled by default (e.g., `..._consensus_stage_Colon.csv`, `..._consensus_stage_Duodenum.csv`).
* Update `commands.md` Step 5 to document stratified consensus runs (auto-loop all organs by default, with/without `--no_self`).

---

## NICE TO HAVE (recommended improvements)

### A) Biopsy-level feature aggregation
* Current pipeline pivots at FOV level (`index="FOV"`) and uses biopsy-grouped CV.
* This overweights biopsies with many FOVs and keeps the feature matrix at FOV granularity.
* Improvement: aggregate FOV rule features into one row per biopsy (mean/max/presence) before modeling.

### B) Confounder adjustment inside each stratum
* Reduce bias from non-rule variables (diagnosis, conditioning, prophylaxis, virus status).
* Options: matched subsets, or add covariates to models.

### C) Within-stratum permutation null baselines
* Shuffle target labels within each organ stratum, rerun evaluation.
* Report observed score vs null distribution to confirm signal is real.

### D) Harmonize simple vs advanced comparisons
* Ensure both scripts use identical row sets per target/organ.
* If they differ, mark comparison as not directly comparable.

### E) Stability-first rule ranking
* Rank rules by cross-fold stability (selection frequency), not only single-run importance.
