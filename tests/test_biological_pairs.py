from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "result_summary"),
               str(ROOT / "result_summary" / "differential_rules")]
import biological_pairs as bp


def test_neighbor_expectation_and_direction_reversal():
    coords = np.array([[0, 0], [1, 0], [10, 0]])
    labels = np.array(["A", "B", "B"])
    result = bp.neighbor_rates(coords, labels, "A", "B", 2)
    reverse = bp.neighbor_rates(coords, labels, "B", "A", 2)
    # A has a B neighbor with certainty under the finite-population benchmark;
    # only one of two B centers has any neighbor, and that neighbor is A.
    assert result["observed_ab"] == 1
    assert result["expected_ab"] == 1
    assert result["observed_ba"] == 0.5
    assert result["expected_ba"] == 0.25
    assert result["difference"] == -reverse["difference"]


def test_unmined_reverse_is_retained_and_ineligible_is_separate():
    raw = pd.DataFrame([dict(FOV="f1", Antecedents="['A_CENTER']",
                             Consequents="['B_NEIGHBOR']", Kind="attracts", Lift=2)])
    cells = pd.DataFrame({"fov": ["f1", "f1", "f2"], "cell type": ["A", "B", "A"]})
    metadata = pd.DataFrame({"FOV": ["f1", "f2"]})
    _, states, eligible, _ = bp.prepare_directional_rules(
        raw, cells, metadata, [dict(a="A", b="B")], min_cells=1)
    assert states.loc["A -> B", "f1"] == 1
    assert states.loc["B -> A", "f1"] == 0
    assert eligible.loc["B -> A", "f1"]
    assert not eligible.loc["B -> A", "f2"]
    pd.testing.assert_series_equal(eligible.loc["A -> B"], eligible.loc["B -> A"], check_names=False)


def test_patient_means_weight_biopsies_equally():
    rows = []
    for biopsy, value in [("b1", 0), ("b1", 0), ("b2", 1)]:
        row = dict(pair="A / B", Organ="Colon", PatientID="p1", Biopsy=biopsy,
                   **{"Clinical score": "Control"})
        for metric in ("observed_ab", "expected_ab", "excess_ab", "observed_ba",
                       "expected_ba", "excess_ba", "difference"):
            row[metric] = value
        rows.append(row)
    result = bp.patient_values(pd.DataFrame(rows), ["Control"])
    assert result["difference"].tolist() == [0.5, 0.5]
    assert result["PatientID"].nunique() == 1


def test_sign_tests_exclude_ties_and_require_independent_patients():
    patients = pd.DataFrame(dict(pair=["A / B"]*6, Organ=["Colon"]*6,
                                 stage=["Control"]*6, excess_ab=[1]*5+[0],
                                 excess_ba=[0]*6, difference=[1]*5+[0]))
    result = bp.sign_evidence(patients, min_patients=5).set_index("endpoint")
    assert result.loc["difference", "nonzero_patients"] == 5
    assert result.loc["difference", "p_value"] == 0.0625
    assert np.isnan(result.loc["excess_ba", "fdr"])


def test_mined_difference_uses_shared_eligible_fields():
    cells = pd.DataFrame({"fov": ["f1", "f1"], "cell type": ["A", "B"],
                          "x_um": [0, 1], "y_um": [0, 0]})
    metadata = pd.DataFrame({"FOV": ["f1"], "Organ": ["Colon"], "Biopsy": ["b1"],
                             "PatientID": ["p1"], "Clinical score": ["Control"]})
    eligible = pd.DataFrame({"f1": [True, True]}, index=["A -> B", "B -> A"])
    states = pd.DataFrame({"f1": [1, -1]}, index=eligible.index)
    diagnostics = bp.spatial_diagnostics(
        cells, metadata, [dict(a="A", b="B", organs=["Colon"])], eligible,
        radius=2, states=states)
    # Attraction difference is 1-0, not the net-state difference 1-(-1).
    assert diagnostics.loc[0, "mined_difference"] == 1
    patients = bp.patient_values(diagnostics, ["Control"])
    assert patients["mined_difference"].tolist() == [1, 1]
    evidence = bp.sign_evidence(patients)
    assert "mined_difference" in evidence.endpoint.tolist()
    assert evidence[evidence.endpoint == "mined_difference"].fdr.isna().all()


def test_directional_display_keeps_each_reverse_adjacent():
    specs = bp.pair_specs([dict(a="A", b="B", organs=["Colon", "Duodenum"])])
    display = bp.directional_specs(specs)
    assert [row["rule"] for row in display] == ["A -> B", "B -> A"] * 2
    assert [row["organ"] for row in display] == ["Colon"] * 2 + ["Duodenum"] * 2


def test_pooled_plot_counts_each_fov_once_and_keeps_nonsignificant_results(monkeypatch):
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")

    metadata = pd.DataFrame({"FOV": ["f1", "f2"], "Organ": ["Colon"] * 2,
                             "Clinical score": ["Control", "Severe"]})
    states = pd.DataFrame([[1, 0]], index=["A -> B"], columns=["f1", "f2"])
    eligible = states.notna()
    rules = pd.DataFrame({"Clean_Rule": ["A -> B"], "FOV": ["f1"], "Lift": [1.5]})
    evidence = pd.DataFrame({"pair": ["A / B"] * 3, "Organ": ["Colon"] * 3,
                             "stage": ["All", "Control", "Severe"],
                             "endpoint": ["mined_difference"] * 3,
                             "fdr": [0.8, np.nan, np.nan],
                             "nonzero_patients": [5, 3, 2], "patients": [8, 4, 4]})
    drawn = []
    monkeypatch.setattr(bp.dv, "_finish", lambda fig, save: drawn.append(fig))
    bp.plot_known_rule(dict(a="A", b="B", pair="A / B", organ="Colon", rule="A -> B"),
                       rules, states, eligible, metadata, evidence, ["Control", "Severe"])
    fig = drawn[0]
    assert fig.axes[0].get_xticklabels()[0].get_text() == "All\n1 | 2"
    assert fig.axes[1].get_xticklabels()[0].get_text() == "All\n1/2"
    assert fig.axes[1].get_xlim() == (-0.5, 2.5)
    assert fig.axes[0].patches[0].get_height() == 50
    assert "FDR" not in fig.texts[1].get_text()
    assert "All = pooled organ" in fig.texts[1].get_text()
    plt.close(fig)
