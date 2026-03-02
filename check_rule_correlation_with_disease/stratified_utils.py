import ast
import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, r2_score
import logging

logger = logging.getLogger("stratified_utils")

# ── Constants ─────────────────────────────────────────────────────────────────

TARGETS = [
    "Pathological stage",
    "GI stage",
    "Cortico Response",
    "Grade GVHD",
    "Survival at follow-up",
    "Clinical score",
    "Pathological score",
]

# Controls are valid for severity targets (control = stage 0),
# but NOT for treatment/outcome targets.
CONTROLS_ELIGIBLE = {
    "Pathological stage": True,
    "GI stage": True,
    "Grade GVHD": True,
    "Clinical score": True,
    "Pathological score": True,
    "Cortico Response": False,
    "Survival at follow-up": False,
}

# ── Data loading ───────────────────────────────────────────────────────────────

def parse_rule_items(item_str):
    """Parse a string-encoded list like "['A', 'B']" into a set of cell-type names."""
    try:
        items = ast.literal_eval(item_str)
        return set(i.replace('_CENTER', '').replace('_NEIGHBOR', '') for i in items)
    except Exception:
        return {str(item_str).strip("[]'\"")}


def load_and_prep_data(input_file, no_self=False):
    """
    Load a rules CSV, apply optional no-self and FDR filters, and return:
    - rule_lift_pivot: FOV × Rule matrix of Lift scores (0 where rule absent)
    - fov_metadata: per-FOV metadata (Biopsy_ID + target columns)
    - raw_rules: the full filtered DataFrame
    Returns (None, None, None) if file missing or no data remains.
    """
    if not os.path.exists(input_file):
        logger.warning(f"File not found: {input_file}")
        return None, None, None

    df = pd.read_csv(input_file)

    if no_self:
        initial_len = len(df)
        def has_overlap(row):
            return not parse_rule_items(row['Antecedents']).isdisjoint(parse_rule_items(row['Consequents']))
        df = df[~df.apply(has_overlap, axis=1)]
        logger.info(f"   Applied No-Self Filter: {initial_len} -> {len(df)} rules")

    if "FDR" in df.columns:
        initial_len = len(df)
        df = df[df["FDR"] < 0.05]
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"   Dropped {dropped} rows with FDR >= 0.05 (Retained: {len(df)})")

    if df.empty:
        logger.warning("   No data left after filtering.")
        return None, None, None

    if "Rule" not in df.columns:
        df["Rule"] = df["Antecedents"] + " -> " + df["Consequents"]

    rule_lift_pivot = df.pivot_table(index="FOV", columns="Rule", values="Lift", aggfunc="mean").fillna(0)

    available_targets = [t for t in TARGETS if t in df.columns]
    fov_metadata = (
        df[["FOV", "Biopsy_ID"] + available_targets]
        .drop_duplicates(subset="FOV")
        .set_index("FOV")
    )

    common = rule_lift_pivot.index.intersection(fov_metadata.index)
    return rule_lift_pivot.loc[common], fov_metadata.loc[common], df


# ── Stratification ─────────────────────────────────────────────────────────────

def filter_viable_stratum(y_meta_subset, target, min_biopsies=3, max_majority_pct=0.80):
    """
    Drop rare classes (< min_biopsies biopsies) from the stratum, then check viability.
    Returns (filtered_y_meta, note) where filtered_y_meta is None if not viable after filtering.
    note describes dropped classes (or None if nothing was dropped).
    """
    biopsy_vals = y_meta_subset[["Biopsy_ID", target]].drop_duplicates()
    biopsy_vals = biopsy_vals[biopsy_vals[target].notna()]
    if len(biopsy_vals) == 0:
        return None, "No valid samples"

    counts = biopsy_vals[target].value_counts()
    rare_classes = counts[counts < min_biopsies].index.tolist()

    if rare_classes:
        rare_biopsies = biopsy_vals[biopsy_vals[target].isin(rare_classes)]["Biopsy_ID"]
        y_filtered = y_meta_subset[~y_meta_subset["Biopsy_ID"].isin(rare_biopsies)]
        biopsy_vals2 = y_filtered[["Biopsy_ID", target]].drop_duplicates()
        biopsy_vals2 = biopsy_vals2[biopsy_vals2[target].notna()]
        counts2 = biopsy_vals2[target].value_counts()
        note = f"Dropped rare class(es) {rare_classes}"
    else:
        y_filtered = y_meta_subset
        counts2 = counts
        note = None

    if len(counts2) < 2:
        return None, f"Only {len(counts2)} class(es) remain after dropping rare classes"

    props2 = counts2 / counts2.sum()
    if props2.max() > max_majority_pct:
        return None, f"Majority class is {props2.max():.1%} after dropping rare classes (need ≤{max_majority_pct:.0%})"

    return y_filtered, note


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_per_class_metrics(true_labels, pred_labels, label_encoder=None):
    """Per-class F1 scores + macro. If label_encoder provided, maps encoded ints back to class names."""
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    result = {}
    for key, val in report.items():
        if key in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        label = label_encoder.inverse_transform([int(key)])[0] if label_encoder else key
        result[str(label)] = round(val['f1-score'], 4)
    result['macro_f1'] = round(report['macro avg']['f1-score'], 4)
    return result


# ── Bootstrap CI ───────────────────────────────────────────────────────────────

def bootstrap_batch_iteration(args):
    """
    Batched bootstrap iteration — top-level function for ProcessPoolExecutor pickling.
    Runs BATCH_SIZE seeds per worker call to amortize pickling cost of large arrays.
    Returns list of macro-F1 (categorical) or R2 (regression) scores, or np.nan on failure.
    """

    X_arr, y_arr, g_arr, unique_groups, is_categorical, seeds = args
    results = []
    
    for seed in seeds:
        rng = np.random.RandomState(seed)

        boot = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        mask = np.isin(g_arr, np.unique(boot))
        X_b, y_b, g_b = X_arr[mask], y_arr[mask], g_arr[mask]

        if len(np.unique(y_b)) < 2 or len(np.unique(g_b)) < 2:
            results.append(np.nan)
            continue

        logo = LeaveOneGroupOut()
        y_true_all, y_pred_all = [], []
        try:
            for train_idx, test_idx in logo.split(X_b, y_b, g_b):
                if len(np.unique(y_b[train_idx])) < 2:
                    continue
                if is_categorical:
                    m = RandomForestClassifier(n_estimators=30, max_depth=5, class_weight='balanced', random_state=seed, n_jobs=1)
                else:
                    m = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=seed, n_jobs=1)
                m.fit(X_b[train_idx], y_b[train_idx])
                y_true_all.extend(y_b[test_idx])
                y_pred_all.extend(m.predict(X_b[test_idx]))
        except Exception:
            results.append(np.nan)
            continue

        if not y_true_all:
            results.append(np.nan)
            continue

        y_t, y_p = np.array(y_true_all), np.array(y_pred_all)
        if is_categorical:
            results.append(float(f1_score(y_t, y_p, average='macro', zero_division=0)))
        else:
            results.append(float(r2_score(y_t, y_p)))

    return results


def run_bootstrap_ci(bootstrap_data, leaderboard_records):
    """
    Run bootstrap CI for all (organ, target) combos and attach CI_Mean/CI_Lower/CI_Upper
    to each record in leaderboard_records in place.
    bootstrap_data: dict of (organ, target) -> (feat_arr, label_arr, group_arr, is_categorical)
    """
    if not bootstrap_data:
        return

    batch_size = 20
    n_bootstrap = 200

    tasks = []
    combo_index = []
    for (organ, target), (feat_arr, label_arr, group_arr, is_categorical) in bootstrap_data.items():
        unique_groups = np.unique(group_arr)
        for batch_start in range(0, n_bootstrap, batch_size):
            seed_batch = list(range(batch_start, min(batch_start + batch_size, n_bootstrap)))
            tasks.append((feat_arr, label_arr, group_arr, unique_groups, is_categorical, seed_batch))
            combo_index.append((organ, target))

    logger.info(f"   >>> Running bootstrap CI ({n_bootstrap} iters batched into {len(tasks)} tasks)...")
    with ProcessPoolExecutor() as executor:
        batch_results = list(executor.map(bootstrap_batch_iteration, tasks))

    ci_scores = {}
    for (organ, target), batch_scores in zip(combo_index, batch_results):
        key = (organ, target)
        if key not in ci_scores:
            ci_scores[key] = []
        for score in batch_scores:
            if not np.isnan(score):
                ci_scores[key].append(score)

    for record in leaderboard_records:
        scores = ci_scores.get((record["Organ"], record["Target"]), [])
        if len(scores) >= 10:
            record["CI_Mean"] = round(float(np.mean(scores)), 4)
            record["CI_Lower"] = round(float(np.percentile(scores, 2.5)), 4)
            record["CI_Upper"] = round(float(np.percentile(scores, 97.5)), 4)
        else:
            record["CI_Mean"] = record["CI_Lower"] = record["CI_Upper"] = np.nan
