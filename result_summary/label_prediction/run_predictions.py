"""Run the predictions below: every setting x model, for every unit, all on every CPU core.

Each run predicts every patient from a model trained on all the other patients. Each
prediction and unit gets its own runs, weights and plot files in output/, saved as soon
as its last run finishes.

    python run_predictions.py          # run, save and plot
    python run_predictions.py --plot   # plot the saved runs again without rerunning
"""
import os
import sys
import time
from collections import Counter

import pandas as pd
from joblib import Parallel, cpu_count, delayed

import prediction_search as ps

CORTICO = dict(target="Cortico Response", value=None, positive="Responder",
               only_patients_with_muscle_and_epithelium=False)
NRM = dict(target="Cause of death", value="NRM", positive=None,
           only_patients_with_muscle_and_epithelium=True)
SURVIVAL = dict(target="Survival at follow-up", value=None, positive="Survived",
                only_patients_with_muscle_and_epithelium=False)
EARLY = dict(target="Days after Transplant grouped", value="<30", positive=None,
             only_patients_with_muscle_and_epithelium=False)
LATE = dict(target="Days after Transplant grouped", value=">100", positive=None,
            only_patients_with_muscle_and_epithelium=False)
DIAGNOSIS = dict(target="Diagnosis", value=None, positive="AML",
                 only_patients_with_muscle_and_epithelium=False)

PREDICTIONS = [CORTICO, NRM, SURVIVAL, EARLY, LATE, DIAGNOSIS]
UNITS = ["patient", "FOV"]


def log(message, pid=None):
    """One line: time, the process that did the work, the message."""
    print(f"{time.strftime('%H:%M:%S')}  pid {pid or os.getpid():>6}  {message}", flush=True)


def all_jobs(cells, fovs, rules):
    """(what the run is, what it needs) for every prediction x unit x setting x model."""
    jobs = []
    for prediction in PREDICTIONS:
        patients = ps.patients_of(prediction, cells, fovs)
        positive, negative = patients["positive"].iloc[0], patients["negative"].iloc[0]
        sides = patients.groupby("Biopsy")["side"].first()
        log(f"--- {ps.name_of(prediction)}: {int(sides.sum())} {positive}, "
            f"{int((sides == 0).sum())} {negative} patients ---")
        for unit in UNITS:
            is_positive, patient_of = ps.labels(patients, unit)
            before = len(jobs)
            for (rule_set, feature, share), table in ps.all_tables(rules, cells, patients, unit):
                for model in ps.models_for(table):
                    about = dict(organ=ps.ORGAN, target=prediction["target"],
                                 positive=positive, negative=negative, unit=unit,
                                 rule_set=rule_set, feature=feature, min_patients=share,
                                 model=model, n_features=table.shape[1])
                    jobs.append(((prediction, about), (table, is_positive, patient_of, model)))
            log(f"[{ps.name_of(prediction)} | {unit}] built {len(jobs) - before} runs")
    return jobs


def group_of(job):
    """The prediction x unit a job belongs to: one set of saved files."""
    (prediction, about), _ = job
    return ps.name_of(prediction), about["unit"]


def run_numbered(i, args):
    """One run, with its place in the job list and the process that ran it."""
    return i, os.getpid(), ps.run_one(*args)


def save_group(jobs, results, group):
    """The runs CSV, weights and plot of one prediction x unit, rows in grid order."""
    rows, weights = [], {}
    for i, job in enumerate(jobs):
        if group_of(job) == group:
            (prediction, about), _ = job
            scores, job_weights = results[i]
            if job_weights is not None:
                weights[len(rows)] = job_weights
            rows.append({**about, **scores})
    ps.save(prediction, about["unit"], pd.DataFrame(rows).round(1), weights)


def run_all(jobs):
    """Every job on every core. A line as each run finishes; a prediction x unit is saved
    as soon as its last run is in."""
    left = Counter(group_of(job) for job in jobs)
    results = {}
    finished = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(run_numbered)(i, args) for i, (_, args) in enumerate(jobs))
    for done, (i, pid, result) in enumerate(finished, 1):
        results[i] = result
        group = group_of(jobs[i])
        about, scores = jobs[i][0][1], result[0]
        share = "" if pd.isna(about["min_patients"]) else about["min_patients"]
        log(f"[{group[0]} | {group[1]}] [{done:>4}/{len(jobs)}]  {about['rule_set']:34} "
            f"{about['feature']:16} {share:>4} {about['model']:26} {about['n_features']:>6} "
            f"features  {scores['balanced_accuracy']:5.1f}%  ({about['positive']} "
            f"{scores['positive_right']:3.0f}%, {about['negative']} "
            f"{scores['negative_right']:3.0f}%)  AUC {scores['auc']:3.0f}", pid)
        left[group] -= 1
        if left[group] == 0:
            log(f"[{group[0]} | {group[1]}] all runs done, saving")
            save_group(jobs, results, group)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        for prediction in PREDICTIONS:
            for unit in UNITS:
                ps.plot_runs(prediction, unit, pd.read_csv(ps.output_path(prediction, "runs", unit)))
    else:
        log(f"=== Label prediction: {len(PREDICTIONS)} predictions x {len(UNITS)} units "
            f"({', '.join(UNITS)}), {cpu_count()} cores ===")
        log("loading data ...")
        start = time.time()
        data = ps.load_data()
        log(f"data loaded in {time.time() - start:.0f} s")
        jobs = all_jobs(*data)
        log(f"=== {len(jobs)} runs, starting ===")
        run_all(jobs)
        log(f"=== all done in {(time.time() - start) / 60:.0f} min ===")
