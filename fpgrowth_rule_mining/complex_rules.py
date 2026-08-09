import pandas as pd
from .transactions import strip_role

ATTRACTS = "attracts"
AVOIDS = "avoids"

def classify_complex_rules(rules, min_lift_gain):
    """
    Adds classification columns to rules.
    Columns added:
    - rule_type: 'pairwise', 'ant-complex', 'con-complex'
    - complex_class: 'new', 'improved', 'redundant', 'consequent-driven', or None
    - simpler_rules: list of simpler rules compared against
    """
    if rules.empty:
        rules = rules.copy()
        rules["rule_type"] = pd.Series(dtype=str)
        rules["complex_class"] = pd.Series(dtype=str)
        rules["simpler_rules"] = pd.Series(dtype=object)
        return rules

    rules = rules.copy()
    
    # Pre-calculate labels for easier lookup
    rules["ant_labels"] = rules["antecedents"].apply(lambda items: frozenset(strip_role(i) for i in items))
    rules["con_labels"] = rules["consequents"].apply(lambda items: frozenset(strip_role(i) for i in items))
    
    # Initialize new columns
    rules["rule_type"] = "pairwise"
    rules["complex_class"] = None
    rules["simpler_rules"] = [[] for _ in range(len(rules))]
    
    # Create a lookup dictionary: (ant_labels, con_labels, kind) -> lift
    lift_map = {}
    for row in rules.itertuples():
        lift_map[(row.ant_labels, row.con_labels, row.kind)] = row.lift

    for idx, row in rules.iterrows():
        ant = row["ant_labels"]
        con = row["con_labels"]
        kind = row["kind"]
        lift = row["lift"]
        
        len_ant = len(ant)
        len_con = len(con)
        total_len = len_ant + len_con
        
        if total_len <= 2:
            continue
            
        if len_ant > 1 and len_con == 1:
            rules.at[idx, "rule_type"] = "ant-complex"
            _classify_type_1(rules, idx, ant, con, kind, lift, lift_map, min_lift_gain)
        elif len_ant == 1 and len_con > 1:
            rules.at[idx, "rule_type"] = "con-complex"
            _classify_type_2(rules, idx, ant, con, kind, lift, lift_map, min_lift_gain)
            
    return rules.drop(columns=["ant_labels", "con_labels"])


def _format_rule(ant, con):
    ant_str = " + ".join(sorted(ant))
    con_str = " + ".join(sorted(con))
    return f"{ant_str} -> {con_str}"


def _classify_type_1(rules, idx, ant, con, kind, lift, lift_map, min_lift_gain):
    # A + B -> C (Compare with A -> C and B -> C)
    simpler_lifts = []
    simpler_rule_strs = []
    
    for a in ant:
        sub_ant = frozenset([a])
        simpler_rule_strs.append(_format_rule(sub_ant, con))
        key = (sub_ant, con, kind)
        if key in lift_map:
            simpler_lifts.append(lift_map[key])
            
    rules.at[idx, "simpler_rules"] = simpler_rule_strs
    
    if not simpler_lifts:
        rules.at[idx, "complex_class"] = "new"
    else:
        max_simpler_lift = max(simpler_lifts)
        
        if kind == AVOIDS:
            beats_it = lift < max_simpler_lift / min_lift_gain
        else:
            beats_it = lift >= max_simpler_lift * min_lift_gain
            
        if beats_it:
            rules.at[idx, "complex_class"] = "improved"
        else:
            rules.at[idx, "complex_class"] = "redundant"


def _classify_type_2(rules, idx, ant, con, kind, lift, lift_map, min_lift_gain):
    # A -> B + C
    
    # 1. Check Consequent-driven (B -> C or C -> B)
    # Are the consequents naturally strongly co-occurring?
    con_lifts = []
    consequent_rule_strs = []
    for c1 in con:
        for c2 in con:
            if c1 != c2:
                c1_set = frozenset([c1])
                c2_set = frozenset([c2])
                consequent_rule_strs.append(_format_rule(c1_set, c2_set))
                # B->C is naturally an attraction rule if they form a niche
                key = (c1_set, c2_set, ATTRACTS)
                if key in lift_map:
                    con_lifts.append(lift_map[key])
                    
    # If B->C or C->B exists and has high lift (e.g. higher than A->B+C)
    if con_lifts and max(con_lifts) >= lift:
        rules.at[idx, "complex_class"] = "consequent-driven"
        rules.at[idx, "simpler_rules"] = consequent_rule_strs
        return

    # 2. Check Niche-defining/Improved or Redundant (A -> B and A -> C)
    simpler_lifts = []
    simpler_rule_strs = []
    
    for c in con:
        sub_con = frozenset([c])
        simpler_rule_strs.append(_format_rule(ant, sub_con))
        key = (ant, sub_con, kind)
        if key in lift_map:
            simpler_lifts.append(lift_map[key])
            
    rules.at[idx, "simpler_rules"] = simpler_rule_strs
    
    if not simpler_lifts:
        rules.at[idx, "complex_class"] = "new"
    else:
        max_simpler_lift = max(simpler_lifts)
        
        if kind == AVOIDS:
            beats_it = lift < max_simpler_lift / min_lift_gain
        else:
            beats_it = lift >= max_simpler_lift * min_lift_gain
            
        if beats_it:
            rules.at[idx, "complex_class"] = "improved"
        else:
            rules.at[idx, "complex_class"] = "redundant"
