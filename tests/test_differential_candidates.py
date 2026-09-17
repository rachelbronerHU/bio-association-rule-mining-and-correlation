from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "result_summary"))
sys.path.insert(0, str(ROOT / "result_summary" / "differential_rules"))

import differential_stats as ds


def test_aggregate_fovs_uses_one_column_per_biopsy():
    values = pd.DataFrame(
        [[1.0, -1.0, 1.0], [np.nan, 0.0, 1.0]],
        index=["a", "b"], columns=["f1", "f2", "f3"],
    )
    metadata = pd.DataFrame({
        "FOV": ["f1", "f2", "f3"],
        "Biopsy": ["b1", "b1", "b2"],
    })

    result = ds.aggregate_fovs(values, metadata)

    assert result.columns.tolist() == ["b1", "b2"]
    assert result.loc["a"].tolist() == [0.0, 1.0]
    assert result.loc["b"].tolist() == [0.0, 1.0]


def test_select_candidates_keeps_statistical_and_large_visible_streams():
    evidence = pd.DataFrame([
        {"rule": "stat", "test": "ordered", "effect": 0.25,
         "abs_effect": 0.25, "p_value": 0.001, "fdr": 0.01,
         "min_n": 4, "steady": True},
        {"rule": "visual", "test": "Control vs Severe", "effect": 0.55,
         "abs_effect": 0.55, "p_value": 0.2, "fdr": 0.4,
         "min_n": 20, "steady": np.nan},
        {"rule": "small", "test": "Control vs Severe", "effect": 0.70,
         "abs_effect": 0.70, "p_value": 0.2, "fdr": 0.4,
         "min_n": 5, "steady": np.nan},
    ])

    result = ds.select_candidates(evidence)

    assert result["rule"].tolist() == ["stat", "visual"]
    assert result.set_index("rule").at["stat", "selection"] == "statistical"
    assert result.set_index("rule").at["visual", "selection"] == "strong visible effect"


def test_select_candidates_can_combine_biopsy_statistics_with_fov_effects():
    biopsy = pd.DataFrame([
        {"rule": "stat", "test": "ordered", "effect": 0.30,
         "abs_effect": 0.30, "p_value": 0.001, "fdr": 0.01,
         "min_n": 4, "steady": True},
        {"rule": "visual", "test": "ordered", "effect": 0.15,
         "abs_effect": 0.15, "p_value": 0.4, "fdr": 0.6,
         "min_n": 4, "steady": True},
    ])
    fov = pd.DataFrame([
        {"rule": "stat", "test": "ordered", "effect": 0.20,
         "abs_effect": 0.20, "p_value": 0.1, "fdr": 0.2,
         "min_n": 20, "steady": True},
        {"rule": "visual", "test": "Control vs Severe", "effect": 0.55,
         "abs_effect": 0.55, "p_value": 0.2, "fdr": 0.4,
         "min_n": 20, "steady": np.nan},
    ])

    result = ds.select_candidates(biopsy, fov)

    assert result["rule"].tolist() == ["stat", "visual"]
    visual = result.set_index("rule").loc["visual"]
    assert visual["selection"] == "strong visible effect"
    assert visual["best_fdr"] == 0.6
    assert visual["best_biopsy_test"] == "ordered"
    assert visual["biopsy_min_n"] == 4


def test_direction_matrix_preserves_ineligible_values():
    states = pd.DataFrame([[1.0, 0.0, -1.0, np.nan]], index=["rule"])

    attraction = ds.direction_matrix(states, 1)

    assert attraction.iloc[0, :3].tolist() == [1.0, 0.0, 0.0]
    assert np.isnan(attraction.iloc[0, 3])


def test_empty_screen_is_valid():
    values = pd.DataFrame([[np.nan, np.nan]], index=["rule"], columns=["f1", "f2"])
    metadata = pd.DataFrame({
        "FOV": ["f1", "f2"],
        "Organ": ["Colon", "Colon"],
        "score": ["Control", "Severe"],
    })

    ordered, pairs = ds.severity_screen(
        values, metadata, "Colon", "score", ["Control", "Severe"],
    )

    assert ordered.empty
    assert all(result.empty for result in pairs.values())
    assert "fdr" in ordered
    assert all("fdr" in result for result in pairs.values())
