"""Load the spatial data and the mined rules, and clean rule names.

Both result-summary notebooks use this so they start from the exact same table.
Keep it simple: this file only reads files and tidies them. No plotting, no analysis.
"""
import os
import ast
import pandas as pd


# ---------------------------------------------------------------------------
# 0. The mining run every notebook reads
# ---------------------------------------------------------------------------

# Change this line and all the notebooks follow. One that wants a different run
# passes its own path instead: dh.load_results(path) or rs.load(path).
RESULT_RUN = 'full_run/binary_CN_4_items_fixed_Epithelial' # "full_run/weighted_fpgrowth_4_items_no_markers_with_ex_per"

# Built from this file's own location, not the working directory: the notebooks sit
# at different depths, so no one relative path would work for all of them.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULT_CSV_PATH = os.path.join(_ROOT, "results", *RESULT_RUN.split("/"), "data")

# A run holds its rules in one of two layouts: split in two by prepare_data.ipynb, or
# a single file already cut down. Which one is on disk says which to read.
PAIRWISE_FILENAME = "results_CN_pairwise.csv"
COMPLEX_FILENAME = "results_CN_complex.csv"
FILTERED_FILENAME = "results_CN_filtered.csv"

# How significant a rule has to be to be worth looking at.
MAX_FDR = 0.05

# Stages left out of everything, by Pathological score. Set here and no notebook sees
# those FOVs at all - not in a heatmap, not in a PCA, not in a cell count.
# Control_S is the second control group, 21 FOVs.
DROP_STAGES = ("Control_S",)         # () keeps everything. One name is fine: "Control_S"

# Cell types left out of every rule. A rule naming one of these is dropped whole, so no
# notebook sees it. Unidentified is not a cell type - it is the cells the classifier
# could not name.
DROP_CELLS = ("Unidentified",)       # () keeps everything. One name is fine: "Unidentified"


# ---------------------------------------------------------------------------
# 1. Load the spatial data (cells, FOVs, biopsies)
# ---------------------------------------------------------------------------

def _find_data_dir():
    """Walk up from the current folder until we find data/MIBIGutCsv."""
    current = os.path.abspath(os.getcwd())
    while current != os.path.dirname(current):
        candidate = os.path.join(current, "data", "MIBIGutCsv")
        if os.path.exists(candidate):
            return candidate
        current = os.path.dirname(current)
    return "../../data/MIBIGutCsv"


def _get_organ(row):
    """Colon or Duodenum for this FOV, taken from the biopsy or the cohort name."""
    if pd.notna(row.get("Localization")):
        return row["Localization"]
    cohort = str(row.get("Cohort", ""))
    if "Colon" in cohort:
        return "Colon"
    if "Duodenum" in cohort:
        return "Duodenum"
    return "Unknown"


# Biopsy columns the notebooks can split rules by, on top of the two scores.
METADATA_COLS = ["Days after Transplant grouped", "Donor type",
                 "Cortico Response", "Survival at follow-up"]


def _control_label(fov):
    """Control FOVs have no biopsy at all; the S_ ones are their own control group."""
    return "Control_S" if str(fov).startswith("S_") else "Control"


def _normalize_coords(row, fov_to_size):
    """Turn pixel coordinates into microns using each FOV's size."""
    size = fov_to_size.get(row["fov"], 800)
    resolution = 1024 if size == 400 else 2048
    return row["centroid_x"] * (size / resolution), row["centroid_y"] * (size / resolution)


def load_spatial_data(data_dir=None):
    """Read the three data files and add the labels the notebooks need.

    Returns (df_cells, df_fovs, df_biopsy):
      - df_cells : one row per cell, with x_um / y_um in microns.
      - df_fovs  : one row per FOV, with Organ, Pathological score, Clinical score,
                   and three counting units: FOV, Biopsy (sample), PatientID (person).
      - df_biopsy: the raw biopsy table.
    """
    if data_dir is None:
        data_dir = _find_data_dir()

    df_cells = pd.read_csv(os.path.join(data_dir, "cell_table.csv"))
    df_fovs = pd.read_csv(os.path.join(data_dir, "fovs_metadata.csv"))
    df_biopsy = pd.read_csv(os.path.join(data_dir, "biopsy_metadata.csv"))

    # Attach the biopsy's scores, location and metadata to each FOV.
    label_cols = ["Pathological score", "Clinical score"] + METADATA_COLS
    df_fovs = df_fovs.merge(
        df_biopsy[["Biopsy_ID", "Localization"] + label_cols],
        left_on="Patient", right_on="Biopsy_ID", how="left",
    )
    df_fovs["Organ"] = df_fovs.apply(_get_organ, axis=1)

    # A FOV with no biopsy row is a control, and gets that label in every column. A FOV
    # that has a biopsy but no value for one column keeps the blank - it is missing, not
    # a control, and the notebook drops those rows for that column.
    is_control = df_fovs["Biopsy_ID"].isna()
    control = df_fovs["FOV"].map(_control_label)
    for col in label_cols:
        df_fovs[col] = df_fovs[col].astype("object").mask(is_control, control)

    # Three ways to count a rule: by FOV, by biopsy (sample), or by patient (person).
    biopsy_to_patient = df_biopsy.set_index("Biopsy_ID")["Patient_ID"].to_dict()
    df_fovs["Biopsy"] = df_fovs["Patient"]  # the 'Patient' column is really a biopsy id
    df_fovs["PatientID"] = df_fovs["Biopsy"].map(biopsy_to_patient).fillna(df_fovs["Biopsy"])

    # Whole stages left out of every notebook, cells and all. One name may be written
    # as a plain string: ("Control_S") is not a tuple, and that is easy to miss.
    if DROP_STAGES:
        drop = [DROP_STAGES] if isinstance(DROP_STAGES, str) else list(DROP_STAGES)
        gone = df_fovs[df_fovs["Pathological score"].isin(drop)]["FOV"]
        df_fovs = df_fovs[~df_fovs["FOV"].isin(gone)]
        df_cells = df_cells[df_cells["fov"].isin(df_fovs["FOV"])]
        print(f"Dropped {gone.nunique()} FOVs: {', '.join(drop)}")

    # Coordinates in microns, so cells from different FOV sizes are comparable.
    fov_to_size = df_fovs.set_index("FOV")["Size [um]"].to_dict()
    df_cells[["x_um", "y_um"]] = df_cells.apply(
        lambda r: pd.Series(_normalize_coords(r, fov_to_size)), axis=1
    )

    print(f"Loaded {len(df_cells)} cells; "
          f"FOVs: {df_fovs['FOV'].nunique()}, "
          f"biopsies: {df_fovs['Biopsy'].nunique()}, "
          f"patients: {df_fovs['PatientID'].nunique()}")
    return df_cells, df_fovs, df_biopsy


# ---------------------------------------------------------------------------
# 2. Load the mined rules (same table for both notebooks)
# ---------------------------------------------------------------------------

def count_items(row):
    """How many items the rule mentions in total (antecedent + consequent).

    Items, not cell types: 'Paneth_CENTER + Paneth_NEIGHBOR' is two. This is the same
    count the library uses for its own Rule_Type, so the two always agree.
    """
    ants = ast.literal_eval(str(row["Antecedents"]))
    cons = ast.literal_eval(str(row["Consequents"]))
    return len(ants) + len(cons)


def _read_rules(directory, rule_max_items):
    """The run's rules, from whichever layout this directory holds.

    Split in two: read the pairwise file, and pay for the complex one only when a
    caller asks for longer rules. A single filtered file holds both, so it is read
    whole and narrowed afterwards.
    """
    pairwise_path = os.path.join(directory, PAIRWISE_FILENAME)
    if os.path.exists(pairwise_path):
        read = [PAIRWISE_FILENAME]
        frames = [pd.read_csv(pairwise_path)]

        complex_path = os.path.join(directory, COMPLEX_FILENAME)
        if rule_max_items > 2 and os.path.exists(complex_path):
            read.append(COMPLEX_FILENAME)
            frames.append(pd.read_csv(complex_path))

        print(f"Read {' + '.join(read)}")
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    filtered_path = os.path.join(directory, FILTERED_FILENAME)
    if os.path.exists(filtered_path):
        print(f"Read {FILTERED_FILENAME}")
        return pd.read_csv(filtered_path)

    print(f"No rules found in {directory}. Run prepare_data.ipynb first.")
    return pd.DataFrame()


def _of_kind(rules, kind):
    """Attraction rules or avoidance rules. An older run has no Kind column, and there a
    rule attracts when its lift is above 1."""
    if "Kind" in rules.columns:
        return rules["Kind"] == kind
    return rules["Lift"] > 1 if kind == "attracts" else rules["Lift"] < 1


def load_results(result_csv_dir=None, rule_max_items=2, kind="attracts", max_fdr=None,
                 informative_only=True):
    """Read the run's rules and keep the ones we want to study.

    result_csv_dir : which run to read; RESULT_CSV_PATH above unless given.
    rule_max_items : keep rules with at most this many items (2 = pairwise).
    kind           : 'attracts', 'avoids', or None for every rule, attracting and avoiding.
    max_fdr        : how significant a rule must be; MAX_FDR above unless given.
                     Applied when the rules carry an Individual_FDR column.
    informative_only : keep only the rules that add something a shorter rule did not
                     already say. Applied when the rules carry an Adds_Information
                     column; pairwise rules all add information, so they never change.
    """
    # Read at call time rather than as a default, so setting dh.MAX_FDR works the way
    # setting dh.RESULT_RUN does.
    max_fdr = MAX_FDR if max_fdr is None else max_fdr
    rules = _read_rules(result_csv_dir or RESULT_CSV_PATH, rule_max_items)
    if rules.empty:
        return rules
    print(f"Loaded {len(rules)} rules.")

    if "Individual_FDR" in rules.columns:
        rules = rules[rules["Individual_FDR"] <= max_fdr].copy()
        print(f"Kept {len(rules)} rules at FDR <= {max_fdr}.")

    if informative_only and "Adds_Information" in rules.columns:
        rules = rules[rules["Adds_Information"].fillna(True).astype(bool)].copy()
        print(f"Kept {len(rules)} rules that add information.")

    if DROP_CELLS:
        drop = {DROP_CELLS} if isinstance(DROP_CELLS, str) else set(DROP_CELLS)
        named = (rules["Antecedents"].apply(base_items)
                 + rules["Consequents"].apply(base_items))
        keep = ~named.apply(lambda items: bool(drop & set(items)))
        rules = rules[keep].copy()
        print(f"Kept {len(rules)} rules naming no {', '.join(sorted(drop))}.")

    rules = rules[rules.apply(count_items, axis=1) <= rule_max_items].copy()
    if kind is not None:
        rules = rules[_of_kind(rules, kind)].copy()
    print(f"Kept {len(rules)} rules (max_items={rule_max_items}, kind={kind}).")
    return rules


# ---------------------------------------------------------------------------
# 3. Clean rule names
# ---------------------------------------------------------------------------

def _strip(items):
    """Drop the _CENTER / _NEIGHBOR tags from a list of stored cell names."""
    return [i.replace("_CENTER", "").replace("_NEIGHBOR", "") for i in items]


def base_items(item_str):
    """The rule's cell types as a list: "['CD4T_CENTER', 'Bcell_NEIGHBOR']" -> ['CD4T', 'Bcell'].

    Use this when you need to count them or check for repeats; use `clean_items` when you
    need the rule's name.
    """
    return _strip(ast.literal_eval(str(item_str)))


def clean_items(item_str):
    """Turn a stored list like "['CD4T_CENTER', 'Bcell_NEIGHBOR']" into "Bcell, CD4T".

    Sorted, so the same rule reads the same everywhere.
    """
    return ", ".join(sorted(base_items(item_str)))


def check_rule_overlap(ant_list, con_list):
    """True if a cell type shows up more than once in the rule (a 'self' rule).

    Catches both 'Muscle -> Muscle' and rules that repeat a type on one side.
    """
    base = _strip(list(ant_list) + list(con_list))
    return len(base) != len(set(base))
