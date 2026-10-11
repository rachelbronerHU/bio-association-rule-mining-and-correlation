"""Is the best rule run real, or luck from picking the best of many runs?

Shuffle the labels among the patients, run the whole rule search on them, and keep its
best score; repeat. If the real best beats almost every shuffled best, it is real.
Each search (real or shuffled) is one job, all on every CPU core.

    python permutation_test.py
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed

import prediction_search as ps
import prediction_vis as pv
from run_predictions import CORTICO, log

PREDICTION = CORTICO
UNIT = "patient"
N_SHUFFLES = 100


def rule_tables(cells, fovs, rules):
    """Every rule table of the search (no references), and each patient's side and name."""
    patients = ps.patients_of(PREDICTION, cells, fovs)
    is_positive, patient_of = ps.labels(patients, UNIT)
    tables = [table for (_, _, share), table in ps.all_tables(rules, cells, patients, UNIT)
              if pd.notna(share)]
    return tables, is_positive, patient_of


def best_of_search(tables, is_positive, patient_of, seed=None):
    """The best % right and best AUC over every rule table x model. With a seed, the
    labels are first shuffled among the patients."""
    if seed is not None:
        is_positive = pd.Series(np.random.default_rng(seed).permutation(is_positive),
                                index=is_positive.index)
    scores = pd.DataFrame(ps.score(table, is_positive, patient_of, model)
                          for table in tables for model in ps.MODELS)
    return {"real": seed is None, "seed": seed, **scores[["balanced_accuracy", "auc"]].max()}


if __name__ == "__main__":
    log(f"=== Permutation test: {ps.name_of(PREDICTION)} | {UNIT}, real labels + "
        f"{N_SHUFFLES} shuffles, {cpu_count()} cores ===")
    start = time.time()
    tables, is_positive, patient_of = rule_tables(*ps.load_data())
    log(f"data loaded; each search is {len(tables)} rule tables x {len(ps.MODELS)} models")

    seeds = [None, *range(N_SHUFFLES)]
    rows = []
    searches = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(best_of_search)(tables, is_positive, patient_of, seed) for seed in seeds)
    for done, row in enumerate(searches, 1):
        rows.append(row)
        labels = "real labels" if row["real"] else f"shuffle {row['seed']}"
        log(f"[{done:>3}/{len(seeds)}] {labels:12} best {row['balanced_accuracy']:5.1f}% right, "
            f"AUC {row['auc']:5.1f}")

    results = pd.DataFrame(rows).sort_values("seed", na_position="first").round(1)
    path = ps.output_path(PREDICTION, "permutation", UNIT)
    results.to_csv(path, index=False)
    log(f"saved {path}")
    for column in ["balanced_accuracy", "auc"]:
        log(f"p-value for best {column}: {ps.p_value(results, column):.3f}")
    pv.plot_permutation(PREDICTION, UNIT)
    log(f"=== all done in {(time.time() - start) / 60:.0f} min ===")
