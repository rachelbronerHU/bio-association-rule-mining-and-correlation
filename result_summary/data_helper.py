"""Load the spatial data and the mined rules, and clean rule names.

Both result-summary notebooks use this so they start from the exact same table.
Keep it simple: this file only reads files and tidies them. No plotting, no analysis.
"""
import os
import ast
import numpy as np
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

# A run's rules, split in two by prepare_data.ipynb.
PAIRWISE_FILENAME = "results_CN_pairwise.csv"
COMPLEX_FILENAME = "results_CN_complex.csv"

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
STAGE_COLS = ["Pathological stage", "GI stage"]


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
    label_cols = ["Pathological score", "Clinical score"] + STAGE_COLS + METADATA_COLS
    df_fovs = df_fovs.merge(
        df_biopsy[["Biopsy_ID", "Localization"] + label_cols],
        left_on="Patient", right_on="Biopsy_ID", how="left",
    )
    df_fovs[STAGE_COLS] = df_fovs[STAGE_COLS].astype("Int64")
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


def fov_fractions(cells):
    """FOV x cell type, each row summing to 1: what each FOV is made of.

    Transpose it for cell types x FOVs.
    """
    counts = cells.groupby(["fov", "cell type"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


# ---------------------------------------------------------------------------
# 2. Load the mined rules (same table for both notebooks)
# ---------------------------------------------------------------------------

PAIRWISE_ITEMS = 2


def n_items(rules):
    """How many items each rule mentions in total (antecedent + consequent).

    Items, not cell types: 'Paneth_CENTER + Paneth_NEIGHBOR' is two.
    """
    return rules["len_ant"] + rules["len_con"]


def rule_shape(rules):
    """'pairwise', 'complex-antecedents' or 'complex-consequents' for each rule."""
    shape = np.select([n_items(rules).eq(PAIRWISE_ITEMS), rules["len_ant"].gt(1)],
                      ["pairwise", "complex-antecedents"], "complex-consequents")
    return pd.Series(shape, index=rules.index)


def _own_test(rules, regular, conditional):
    """The regular test's column for pairwise rules, the conditional test's for complex ones."""
    return rules[regular].where(n_items(rules).eq(PAIRWISE_ITEMS), rules[conditional])


def rule_p_value(rules):
    """Each rule's own p-value."""
    return _own_test(rules, "Regular_P_Value", "conditional_p_value")


def rule_fdr(rules):
    """Each rule's own FDR."""
    return _own_test(rules, "Individual_FDR", "conditional_individual_fdr")


def _read_rules(directory, rule_max_items):
    """The run's rules: the pairwise file, and the complex one only when a caller asks
    for longer rules."""
    pairwise_path = os.path.join(directory, PAIRWISE_FILENAME)
    if not os.path.exists(pairwise_path):
        print(f"No rules found in {directory}. Run prepare_data.ipynb first.")
        return pd.DataFrame()

    read = [PAIRWISE_FILENAME]
    frames = [pd.read_csv(pairwise_path)]

    complex_path = os.path.join(directory, COMPLEX_FILENAME)
    if rule_max_items > PAIRWISE_ITEMS and os.path.exists(complex_path):
        read.append(COMPLEX_FILENAME)
        frames.append(pd.read_csv(complex_path))

    print(f"Read {' + '.join(read)}")
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def load_results(result_csv_dir=None, rule_max_items=PAIRWISE_ITEMS, kind="attracts", max_fdr=None):
    """Read the run's rules and keep the ones we want to study.

    result_csv_dir : which run to read; RESULT_CSV_PATH above unless given.
    rule_max_items : keep rules with at most this many items (PAIRWISE_ITEMS = pairwise).
    kind           : 'attracts', 'avoids', or None for every rule, attracting and avoiding.
    max_fdr        : how significant a rule must be; MAX_FDR above unless given.
                     Complex rules are held to it by their conditional test.
    """
    # Read at call time rather than as a default, so setting dh.MAX_FDR works the way
    # setting dh.RESULT_RUN does.
    max_fdr = MAX_FDR if max_fdr is None else max_fdr
    rules = _read_rules(result_csv_dir or RESULT_CSV_PATH, rule_max_items)
    if rules.empty:
        return rules
    print(f"Loaded {len(rules)} rules.")

    rules = rules[rule_fdr(rules) <= max_fdr].copy()
    print(f"Kept {len(rules)} rules at FDR <= {max_fdr}.")

    if DROP_CELLS:
        drop = {DROP_CELLS} if isinstance(DROP_CELLS, str) else set(DROP_CELLS)
        named = (rules["Antecedents"].apply(base_items)
                 + rules["Consequents"].apply(base_items))
        keep = ~named.apply(lambda items: bool(drop & set(items)))
        rules = rules[keep].copy()
        print(f"Kept {len(rules)} rules naming no {', '.join(sorted(drop))}.")

    rules = rules[n_items(rules) <= rule_max_items].copy()
    if kind is not None:
        rules = rules[rules["Kind"] == kind].copy()
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


def prepare_rules(results, no_self=True):
    """Clean rule names and encode attraction (+1) or avoidance (-1)."""
    rules = results.copy()
    if no_self:
        repeated = rules.apply(
            lambda row: check_rule_overlap(
                base_items(row["Antecedents"]), base_items(row["Consequents"])
            ),
            axis=1,
        )
        rules = rules.loc[~repeated].copy()

    rules["Clean_Ant"] = rules["Antecedents"].apply(clean_items)
    rules["Clean_Con"] = rules["Consequents"].apply(clean_items)
    rules["Clean_Rule"] = rules["Clean_Ant"] + " -> " + rules["Clean_Con"]
    kind = (
        rules["Kind"].astype(str).str.lower()
        if "Kind" in rules
        else pd.Series("", index=rules.index)
    )
    rules["state"] = np.select(
        [kind.str.startswith("attract"), kind.str.startswith("avoid")],
        [1, -1],
        default=np.where(rules["Lift"] > 1, 1, -1),
    )
    return rules


MISSING_POPULATION = {"Muscle": "Stroma", "BrunnerGland": "Epithel"}


def rule_class(rules, df_cells):
    """Which tissue parts each rule joins, from the population of its cell types:
    Immune–Immune, Immune–Epithel, Immune–Stroma, or Tissue structure (no immune cells)."""
    population = (df_cells.groupby("cell type")["population"].first()
                  .fillna(pd.Series(MISSING_POPULATION)))
    def one(rule):
        parts = set(population[rule.replace(" -> ", ", ").split(", ")])
        if "Immune" not in parts:
            return "Tissue structure"
        return "Immune–" + ("–".join(sorted(parts - {"Immune"})) or "Immune")
    return pd.Series([one(rule) for rule in rules], index=rules)
