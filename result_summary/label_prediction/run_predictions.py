"""Run the predictions below: every setting x model, for every unit, all on every CPU core.

Each run predicts every patient from a model trained on all the other patients. Each
prediction and unit gets its own runs, weights and plot files in output/.

    python run_predictions.py          # run, save and plot
    python run_predictions.py --plot   # plot the saved runs again without rerunning
"""
import sys
import time

import pandas as pd
from joblib import Parallel, delayed

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


def all_jobs(cells, fovs, rules):
    """(what the run is, what it needs) for every prediction x unit x setting x model."""
    jobs = []
    for prediction in PREDICTIONS:
        patients = ps.patients_of(prediction, cells, fovs)
        for unit in UNITS:
            is_positive, patient_of = ps.labels(patients, unit)
            for setting, table in ps.all_tables(rules, cells, patients, unit):
                for model in ps.models_for(table):
                    about = dict(organ=ps.ORGAN, target=prediction["target"],
                                 positive=patients["positive"].iloc[0],
                                 negative=patients["negative"].iloc[0], unit=unit,
                                 rule_set=setting[0], feature=setting[1],
                                 min_patients=setting[2], model=model,
                                 n_features=table.shape[1])
                    jobs.append(((prediction, about), (table, is_positive, patient_of, model)))
    return jobs


def run_numbered(i, args):
    """One run, with its place in the job list so it can be put back in order."""
    return i, ps.run_one(*args)


def run_all(jobs):
    """Every job on every core; one line printed as each finishes. Results in job order."""
    results = [None] * len(jobs)
    start = time.time()
    finished = Parallel(n_jobs=-1, return_as="generator_unordered")(
        delayed(run_numbered)(i, args) for i, (_, args) in enumerate(jobs))
    for done, (i, result) in enumerate(finished, 1):
        results[i] = result
        about, scores = jobs[i][0][1], result[0]
        minutes, seconds = divmod(int(time.time() - start), 60)
        print(f"[{done:>4}/{len(jobs)}  {minutes:>3}:{seconds:02}]  {about['target']:16} "
              f"{about['unit']:8} {about['rule_set']:34} {about['feature']:16} "
              f"{'' if pd.isna(about['min_patients']) else about['min_patients']:>4} {about['model']:26} {about['n_features']:>6} features  "
              f"{scores['balanced_accuracy']:5.1f}%  ({about['positive']} "
              f"{scores['positive_right']:3.0f}%, {about['negative']} "
              f"{scores['negative_right']:3.0f}%)  AUC {scores['auc']:3.0f}")
    return results


def save_all(jobs, results):
    """One runs CSV, weights file and plot per prediction x unit."""
    for prediction in PREDICTIONS:
        for unit in UNITS:
            rows, weights = [], {}
            for ((job_prediction, about), _), (scores, job_weights) in zip(jobs, results):
                if job_prediction is prediction and about["unit"] == unit:
                    if job_weights is not None:
                        weights[len(rows)] = job_weights
                    rows.append({**about, **scores})
            ps.save(prediction, unit, pd.DataFrame(rows).round(1), weights)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        for prediction in PREDICTIONS:
            for unit in UNITS:
                ps.plot_runs(prediction, unit, pd.read_csv(ps.output_path(prediction, "runs", unit)))
    else:
        jobs = all_jobs(*ps.load_data())
        save_all(jobs, run_all(jobs))
