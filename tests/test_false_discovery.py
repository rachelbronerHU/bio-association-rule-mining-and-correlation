"""
Tests for the dataset-level claim.

The counts are built by hand so that n and k are known before anything runs.
"""

import pandas as pd
import pytest

from fpgrowth_rule_mining.runner import RunReport, SampleResult
from fpgrowth_rule_mining.validation.false_discovery import (
    false_discovery_rates,
    group_p_value,
    recurrence_p_value,
)

RULE = (("A_CENTER",), ("B_NEIGHBOR",))
OTHER = (("C_CENTER",), ("D_NEIGHBOR",))


def sample(sample_id, p_by_rule, eligible, survived=None):
    """
    One mined sample: which rules were found, with what p, and what had enough cells.

    survived lists the rules that add information. It defaults to all of them, since
    most tests do not care about that column.
    """
    def frame_of(rules):
        rows = [{"antecedents": a, "consequents": c, "p_value": p_by_rule[(a, c)]} for a, c in rules]
        return pd.DataFrame(rows, columns=["antecedents", "consequents", "p_value"])

    tested = frame_of(p_by_rule)
    adds = set(p_by_rule if survived is None else survived)
    kept = tested.copy()
    kept["adds_information"] = [(a, c) in adds
                                for a, c in zip(kept["antecedents"], kept["consequents"])]
    return SampleResult(sample_id, kept, tested, {"labels_with_enough_cells": frozenset(eligible)})


def report(*samples):
    return RunReport(results=list(samples))


# --- the three steps on their own ---------------------------------------------

def test_group_p_value_charges_for_the_number_of_attempts():
    """A patient with eight FOVs had eight chances at a lucky one, so eight times the bar."""
    assert group_p_value([0.01, 0.4, 0.9], attempts=3) == pytest.approx(0.03)
    assert group_p_value([0.5], attempts=1) == pytest.approx(0.5)
    assert group_p_value([0.4, 0.9], attempts=2) == 0.8
    assert group_p_value([0.9], attempts=8) == 1.0, "it can never exceed 1"


def test_recurrence_p_value_reads_as_more_often_than_chance():
    many = recurrence_p_value(passed=40, tested=218, alpha=0.05)   # chance predicts ~11
    few = recurrence_p_value(passed=11, tested=218, alpha=0.05)    # exactly what chance predicts
    assert many < 0.001
    assert few > 0.4
    assert recurrence_p_value(passed=0, tested=0, alpha=0.05) == 1.0


def test_false_discovery_rates_are_not_simply_stricter_when_wider():
    """BH divides by rank as well as count, so a wider family can lower an answer."""
    narrow = false_discovery_rates([0.04] + [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])[0]
    wide = false_discovery_rates([0.04] + [0.001] * 700 + [0.5] * 299)[0]
    assert wide < narrow


def test_the_correction_only_ever_raises_a_p_value():
    """Within one family, no rule ends up looking better than its raw p-value."""
    raw = [0.01, 0.02]
    corrected = false_discovery_rates(raw)
    assert (corrected >= raw).all()
    assert corrected[-1] == pytest.approx(0.02), "the largest is never raised"


# --- the whole claim -----------------------------------------------------------

def test_without_groups_it_is_plain_counting():
    """groups=None must reduce exactly to one vote per sample, with no penalty."""
    found = report(*[sample(f"s{i}", {RULE: 0.001}, "AB") for i in range(10)])
    row = found.dataset_significance(groups=None).iloc[0]

    assert row["groups_tested"] == 10
    assert row["groups_passed"] == 10
    assert row["p_value"] < 1e-9


def test_ten_fovs_from_one_patient_count_once():
    """The whole point of grouping: one patient must not outvote everybody else."""
    samples = [sample(f"s{i}", {RULE: 0.001}, "AB") for i in range(10)]
    groups = {f"s{i}": "patient_1" for i in range(10)}

    ungrouped = report(*samples).dataset_significance(groups=None).iloc[0]
    grouped = report(*samples).dataset_significance(groups=groups).iloc[0]

    assert ungrouped["groups_tested"] == 10
    assert grouped["groups_tested"] == 1, "ten FOVs from one patient are one piece of evidence"
    assert grouped["p_value"] > ungrouped["p_value"], "one patient must be weaker evidence than ten"


def test_a_sample_that_could_not_see_the_rule_is_not_an_attempt():
    """A FOV with no B cells never sat the exam; it must not count as a failure."""
    can_see = [sample(f"yes{i}", {RULE: 0.001}, "AB") for i in range(5)]
    cannot = [sample(f"no{i}", {}, "AC") for i in range(20)]        # no B label at all

    row = report(*can_see, *cannot).dataset_significance().iloc[0]
    assert row["groups_tested"] == 5, "the 20 samples without B are not attempts"
    assert row["groups_passed"] == 5


def test_testable_but_absent_counts_as_a_failure():
    """The rule could have been found here and was not. That is evidence against it."""
    found = [sample(f"yes{i}", {RULE: 0.001}, "AB") for i in range(3)]
    silent = [sample(f"no{i}", {OTHER: 0.001}, "ABCD") for i in range(7)]   # B was eligible

    row = report(*found, *silent).dataset_significance()
    rule_row = row[row["antecedents"] == RULE[0]].iloc[0]
    assert rule_row["groups_tested"] == 10, "all ten could have produced it"
    assert rule_row["groups_passed"] == 3, "seven had the chance and did not"


def test_rules_keep_their_centre_and_neighbour_roles():
    """
    Two rules naming the same pair with the centre swapped are different claims and
    must not be pooled.
    """
    swapped = (("B_CENTER",), ("A_NEIGHBOR",))
    found = report(*[sample(f"s{i}", {RULE: 0.001, swapped: 0.5}, "AB") for i in range(6)])

    result = found.dataset_significance()
    assert len(result) == 2, "the two directions must stay separate"


def test_a_real_finding_clears_the_cutoff_and_noise_does_not():
    """
    The whole chain, end to end: one rule that genuinely recurs must come out under
    a 0.05 cutoff, and rules that are only noise must not — even though the noise
    rules outnumber it fifty to one and drag the correction with them.
    """
    real = (("A_CENTER",), ("B_NEIGHBOR",))
    noise = [((f"N{i}_CENTER",), ("B_NEIGHBOR",)) for i in range(50)]

    samples = []
    for s in range(20):
        p_by_rule = {real: 0.001}                      # holds strongly everywhere
        p_by_rule.update({rule: 0.5 for rule in noise})  # never convincing anywhere
        samples.append(sample(f"s{s}", p_by_rule, "AB" + "".join(f"N{i}" for i in range(50))))

    result = report(*samples).dataset_significance(alpha=0.05)
    is_real = result["antecedents"] == real[0]
    found, noisy = result[is_real].iloc[0], result[~is_real]

    assert len(result) == 51
    assert found["groups_passed"] == 20
    assert found["dataset_fdr"] <= 0.05, "a rule holding in all 20 groups must survive"

    assert (noisy["groups_passed"] == 0).all()
    assert (noisy["dataset_fdr"] > 0.05).all(), "rules that never convinced anyone must not survive"


def test_a_rule_redundant_everywhere_is_not_asked_about():
    """A rule a shorter one already said in every sample is not a separate claim."""
    always_redundant = report(*[
        sample(f"s{i}", {RULE: 0.001, OTHER: 0.001}, "ABCD", survived=[RULE])
        for i in range(5)
    ])
    result = always_redundant.dataset_significance()

    assert len(result) == 1, "only the rule that survived somewhere gets a test"
    assert result.iloc[0]["antecedents"] == RULE[0]


def test_a_rule_filtered_in_one_sample_keeps_its_evidence_from_the_others():
    """
    The two frames doing two jobs. Filtered in sample 0, kept in the rest — so it is a
    claim, and all five samples still count toward it. Reading the counts from the
    filtered frame instead would turn sample 0 into a failure it never was.
    """
    samples = [sample("s0", {RULE: 0.001}, "AB", survived=[])] + [
        sample(f"s{i}", {RULE: 0.001}, "AB") for i in range(1, 5)
    ]
    row = report(*samples).dataset_significance().iloc[0]

    assert row["groups_tested"] == 5, "the sample where it was filtered still counts"
    assert row["groups_passed"] == 5, "and its p-value there still counts as a pass"


def test_with_nothing_marked_redundant_every_rule_is_asked_about():
    """When every rule adds information, all of them are claims."""
    found = report(*[sample(f"s{i}", {RULE: 0.001, OTHER: 0.5}, "ABCD") for i in range(5)])
    assert len(found.dataset_significance()) == 2


def test_empty_report_gives_an_empty_answer():
    assert report().dataset_significance().empty


def test_a_sample_missing_its_label_record_stops_the_run():
    """
    The worst failure this code could have: without the record of which labels a
    sample could speak about, every rule looks untestable and the run reports
    "nothing is significant" — a wrong answer that looks exactly like a real one.
    It must stop instead.
    """
    frame = pd.DataFrame([{"antecedents": RULE[0], "consequents": RULE[1], "p_value": 0.001,
                           "adds_information": True}])
    broken = SampleResult("s0", frame, frame, stats={})     # no labels recorded

    with pytest.raises(KeyError):
        report(broken).dataset_significance()
