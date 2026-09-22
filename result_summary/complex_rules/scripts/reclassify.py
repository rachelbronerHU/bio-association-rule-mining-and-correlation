"""The new three-item classification, under test. The mining package is untouched.

Takes a prepared rule table (ci.prepare) and adds three columns, all ending in _2 so the
stored ones stay readable beside them. Longer and pairwise rules keep an empty class.

A three-item rule is weighed against the two-item rules in its own field that carry the
same arrow with one item dropped from its longer side. Only a shorter rule that keeps
the mining center was measured on the same patches, so only those are used. Multiple
antecedents go through Webb's productive test, multiple consequents through conviction.
Avoidance reads upside down: there the smaller value is the stronger one.
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

NOT_IMPROVING = 'not improving'
STRONGER = 'stronger than shorter'
NO_SHORTER = 'no shorter'
SIMPLER_ARE_NOISE = 'simpler_are_noise'

ANT, CON = 'ant-complex', 'con-complex'


def transaction_counts(run_dir):
    """How many patches each field was mined from, read from the run log."""
    log = (Path(run_dir) / 'run.log').read_text()
    return {fov: int(n) for fov, n in re.findall(r'\[(.+?)\] (\d+) transactions', log)}


def reclassify(rules, transactions, max_fdr=.05, alpha=.05, min_lift_gain=1.1):
    """Add Complex_Class_2, Complex_Class_3, Adds_Information_2 and Simpler_Rules_2."""
    rules = rules.copy()
    table = _table(rules, transactions, max_fdr)
    shorter = _shorter_lookup(table)

    found = [[parent for key in _shorter_keys(row.key) for parent in shorter[key]]
             for row in table.itertuples()]
    rules['Simpler_Rules_2'] = [[parent.name for parent in group] for group in found]
    rules['Complex_Class_2'] = [_class_of(row, group, alpha)
                                for row, group in zip(table.itertuples(), found)]
    rules['Complex_Class_3'] = [_class_with_lift(row, group, verdict, min_lift_gain)
                                for row, group, verdict in zip(table.itertuples(), found,
                                                               rules.Complex_Class_2)]
    rules['Adds_Information_2'] = rules.Complex_Class_2 != NOT_IMPROVING
    return rules


def _table(rules, transactions, max_fdr):
    """The few numbers the comparison needs, one row per stored occurrence."""
    return pd.DataFrame({
        'key': list(zip(rules.FOV, rules.Kind, rules.ant, rules.con)),
        'name': rules.stored_rule.to_numpy(),
        'kind': rules.Kind.to_numpy(),
        'type': rules.Rule_Type.to_numpy(),
        'n_items': rules.n_items.to_numpy(),
        'support': rules.Support.to_numpy(),
        'lift': rules.Lift.to_numpy(),
        # The antecedent side on its own: expected support is the two sides multiplied.
        'ant_support': (rules.Expected_support / _consequent_support(rules)).to_numpy(),
        'confidence': rules.Confidence.to_numpy(),
        'conviction': rules.Conviction.to_numpy(),
        'patches': rules.FOV.map(transactions).to_numpy(),
        'passed': (rules.Individual_FDR <= max_fdr).to_numpy(),
    })


def _consequent_support(rules):
    """The consequent side on its own; zero-lift avoidance reads it from conviction."""
    return (rules.Confidence / rules.Lift.replace(0, np.nan)).fillna(1 - rules.Conviction)


def _shorter_lookup(table):
    """Every two-item rule, by field, direction and exact items."""
    lookup = defaultdict(list)
    for row in table[table.n_items == 2].itertuples():
        lookup[row.key].append(row)
    return lookup


def _shorter_keys(key):
    """The same rule with one item dropped from its longer side, the center kept."""
    fov, kind, ant, con = key
    shorter = ([(fov, kind, (one,), con) for one in ant] if len(ant) == 2
               else [(fov, kind, ant, (one,)) for one in con])
    return [key for key in shorter if any('_CENTER' in item for item in key[2] + key[3])]


def _class_of(row, shorter, alpha):
    """Where one rule stands next to the shorter rules in its own field."""
    if row.n_items != 3:
        return None
    if not shorter:
        return NO_SHORTER
    judged = [parent for parent in shorter if parent.passed]
    if not judged:
        return SIMPLER_ARE_NOISE
    beats = _beats_confidence if row.type == ANT else _beats_conviction
    return STRONGER if all(beats(row, parent, alpha) for parent in judged) else NOT_IMPROVING


def _class_with_lift(row, shorter, verdict, min_lift_gain):
    """Multiple antecedents: Webb's test, and lift at least min_lift_gain past every shorter rule."""
    if row.type != ANT or verdict != STRONGER:
        return verdict
    judged = [parent for parent in shorter if parent.passed]
    far = all(row.lift >= parent.lift * min_lift_gain if row.kind == 'attracts'
              else row.lift < parent.lift / min_lift_gain for parent in judged)
    return STRONGER if far else NOT_IMPROVING


def _improved(rule, parent, value):
    """The gap between a rule and a shorter one, in the rule's own direction."""
    gap = getattr(rule, value) - getattr(parent, value)
    return gap > 0 if rule.kind == 'attracts' else gap < 0


def _beats_conviction(rule, parent, alpha):
    """Multiple consequents: conviction further from 1 than the shorter rule's."""
    return _improved(rule, parent, 'conviction')


def _beats_confidence(rule, parent, alpha):
    """Multiple antecedents: confidence improved, and not by chance (Webb's test)."""
    return _improved(rule, parent, 'confidence') and _improvement_p(rule, parent) <= alpha


def _improvement_p(rule, parent):
    """Fisher's exact test over the four patch counts, added item present or absent."""
    both = round(rule.support * rule.patches)
    antecedents = round(rule.ant_support * rule.patches)
    shorter_both = round(parent.support * parent.patches)
    shorter_antecedent = round(parent.ant_support * parent.patches)
    table = [[both, antecedents - both],
             [shorter_both - both,
              shorter_antecedent - antecedents - shorter_both + both]]
    side = 'greater' if rule.kind == 'attracts' else 'less'
    return fisher_exact(np.clip(table, 0, None), alternative=side).pvalue
