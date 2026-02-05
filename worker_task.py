import pandas as pd
import numpy as np
import time
import logging
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from statsmodels.stats.multitest import multipletests
import transactions as tx

# Setup Worker Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("worker")

def select_top_rules(rules, n=2000):
    if len(rules) <= n: return rules
    half = n // 2
    top_pos = rules.sort_values("lift", ascending=False).head(half)
    top_neg = rules.sort_values("lift", ascending=True).head(half)
    return pd.concat([top_pos, top_neg]).drop_duplicates()

def mine_rules_fpgrowth(transactions, config, method="BAG", sample_id: str = None):
    if not transactions: return pd.DataFrame(), 0
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)
    
    frequent_itemsets = fpgrowth(df_trans, min_support=config["MIN_SUPPORT"], use_colnames=True)
    if frequent_itemsets.empty: return pd.DataFrame(), 0
        
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=config["MIN_CONFIDENCE"])
    
    # Ensure conviction is present (mlxtend usually adds it, but we handle inf explicitly if needed or missing)
    if "conviction" not in rules.columns:
        # Avoid division by zero
        # conviction = (1 - support_cons) / (1 - confidence)
        # if confidence == 1, conviction = inf
        with np.errstate(divide='ignore', invalid='ignore'):
            rules["conviction"] = (1 - rules["consequent support"]) / (1 - rules["confidence"])
            rules["conviction"] = rules["conviction"].fillna(np.inf)

    # Filter by Lift, Leverage, and Conviction
    is_valid = _filter_rule_by_scores_mask(config, rules["lift"], rules["leverage"], rules["conviction"])
    
    rules = rules[is_valid]
    
    # Snapshot of Raw Rules (valid scores, before structural/redundancy filters)
    raw_rules = rules.copy() if not rules.empty else pd.DataFrame()
    
    if not rules.empty:
        rules["len_ant"] = rules["antecedents"].apply(len)
        rules["len_con"] = rules["consequents"].apply(len)
        rules = rules[(rules["len_ant"] + rules["len_con"]) <= config["MAX_RULE_LENGTH"]]
        
        if method in ["CN", "KNN_R"]:
            rules["has_center_ant"] = rules["antecedents"].apply(lambda x: any("_CENTER" in item for item in x))
            rules = rules[rules["has_center_ant"]].drop(columns=["has_center_ant"])

        if rules.empty: return pd.DataFrame(), 0, raw_rules

        rules["antecedents"] = rules["antecedents"].apply(lambda x: tuple(sorted(list(x))))
        rules["consequents"] = rules["consequents"].apply(lambda x: tuple(sorted(list(x))))
        # rules["rule_str"] = rules.apply(lambda x: f"{' + '.join(x['antecedents'])} -> {' + '.join(x['consequents'])}", axis=1)
        
        # --- Redundancy Filter ---
        rules, n_removed = _filter_redundant_rules(rules, config)
        
        rules = select_top_rules(rules, n=2000)
        return rules, n_removed, raw_rules
        
    return rules, 0, raw_rules

def check_rule_in_transactions(rule_row, transactions, config):
    """
    Checks if a specific rule passes thresholds in a given transaction set.
    Instead of running FP-Growth, we compute support/confidence/lift manually.
    """
    if not transactions: return False
    
    N = len(transactions)
    antecedents = set(rule_row["antecedents"])
    consequents = set(rule_row["consequents"])
    both = antecedents | consequents
    
    count_ant = 0
    count_both = 0
    count_cons = 0 # needed for lift
    
    # Single pass
    for t in transactions:
        t_set = set(t)
        has_ant = antecedents.issubset(t_set)
        has_cons = consequents.issubset(t_set)
        
        if has_ant: count_ant += 1
        if has_cons: count_cons += 1
        if has_ant and has_cons: count_both += 1
            
    support_both = count_both / N
    support_ant = count_ant / N
    support_cons = count_cons / N
    
    if support_both < config["MIN_SUPPORT"]: return False
    
    if support_ant == 0: return False # Should not happen if support_both > 0
    
    confidence = support_both / support_ant
    if confidence < config["MIN_CONFIDENCE"]: return False
    
    if support_cons == 0: return False
    
    lift = confidence / support_cons
    leverage = support_both - (support_ant * support_cons)
    
    if confidence >= 1.0:
        conviction = float('inf')
    else:
        conviction = (1 - support_cons) / (1 - confidence)

    # Check if it meets the criteria
    # Note: We check if it is a 'valid discovery' in this random set
    # Using the same criteria as mining
    
    return _filter_rule_by_scores_mask(config, lift, leverage, conviction)

def validate_rules_exact(neighborhoods, cell_types, rules_df, config, method):
    """
    Validates rules by:
    1. Permuting cell types
    2. Re-building transactions (applying dominance filter dynamically)
    3. Checking if rule would be 'mined' (meets thresholds)
    """
    if rules_df.empty: return rules_df
    
    n_perms = config["N_PERMUTATIONS"]
    p_values = []
    
    # Cache original results (already checked by mining, but good to have baseline)
    # We assume rules_df contains rules that PASSED originally.
    
    # We need to perform n_perms
    # To optimize: we can batch verify all rules against one permutation
    
    better_counts = np.zeros(len(rules_df))
    
    for _ in range(n_perms):
        # 1. Shuffle
        shuffled_types = np.random.permutation(cell_types)
        
        # 2. Rebuild Transactions
        # Uses the PRE-CALCULATED neighborhoods structure
        perm_trans, _ = tx.build_transactions_from_neighborhoods(neighborhoods, shuffled_types, method)
        
        # 3. Check each rule
        for idx, (i, row) in enumerate(rules_df.iterrows()):
            if check_rule_in_transactions(row, perm_trans, config):
                better_counts[idx] += 1
                
    p_values = (better_counts + 1) / (n_perms + 1)
    rules_df["p_value"] = p_values
    rules_df["p_value_adj"] = _apply_fdr_correction(p_values)
    return rules_df

def process_single_sample(sample_id, group_val, df_sample, method, config):
    pid = os.getpid()
    prefix = f"[Worker {pid}] [Sample {sample_id}]"
    
    coords = df_sample[["x", "y"]].values
    cell_types = df_sample["cell_type"].values
    
    # 1. Get Structure (Neighborhoods) - ONE TIME
    t0 = time.time()
    neighborhoods = tx.get_neighborhoods(coords, method, config)
    t_struct = time.time() - t0
    
    # 2. Get Initial Transactions
    t0 = time.time()
    trans, stats = tx.build_transactions_from_neighborhoods(neighborhoods, cell_types, method)
    t_trans = time.time() - t0
    
    # 3. Mine
    t0 = time.time()
    rules, n_removed, raw_rules = mine_rules_fpgrowth(trans, config, method=method, sample_id=sample_id)
    t_mine = time.time() - t0
    n_raw = len(rules)
    
    if stats is None: stats = {}
    stats["redundant_removed"] = n_removed
    
    # 4. Validate (Exact Re-check)
    n_val = 0
    rules_v = pd.DataFrame()
    t_val = 0.0
    
    if not rules.empty:
        if len(rules) > 2000: rules = select_top_rules(rules, n=2000)
        
        t0 = time.time()
        rules_v = validate_rules_exact(neighborhoods, cell_types, rules, config, method=method)
        t_val = time.time() - t0
        n_val = len(rules_v)
        
    logger.info(f"{prefix} {method}: {len(trans)} Trans | {n_raw} Rules Found | {n_val} Validated (S:{t_struct:.2f}s T:{t_trans:.2f}s V:{t_val:.2f}s)")
    
    return {
        "Sample": sample_id,
        "Group": group_val,
        "Rules": rules_v,
        "RawRules": raw_rules,
        "Stats": stats
    }

def _filter_rule_by_scores_mask(config, lift, leverage, conviction):
    min_conv = config.get("MIN_CONVICTION", 1.0)
    is_positive = (lift >= config["MIN_LIFT"]) & (leverage >= config["MIN_LEVERAGE"]) & (conviction >= min_conv)
    is_negative = (lift <= config["MAX_NEGATIVE_LIFT"])
    return is_positive | is_negative

def _apply_fdr_correction(p_values):
    """
    Applies Benjamini-Hochberg correction to p-values.
    Returns adjusted p-values.
    """
    if len(p_values) == 0: return []
    reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    return pvals_corrected

def _filter_redundant_rules(rules, config):
    """
    Filters redundant rules based on Occam's Razor.
    A rule is redundant if a simpler rule (subset antecedent) exists with similar or better lift.
    Reference: https://www.bayardo.org/ps/icde99.pdf
    """
    if rules.empty: return rules, 0
    
    threshold_factor = config.get("MIN_REDUNDANCY_LIFT_IMPROVEMENT", 1.1)
    indices_to_drop = set()
    
    # Pre-process: Create frozensets for antecedents once
    rules['ant_frozen'] = rules['antecedents'].apply(frozenset)
    
    # Group by Consequent (we only compare rules predicting the same thing)
    grouped = rules.groupby('consequents')
    
    for _, group in grouped:
        # Sort by antecedent length (shortest first)
        sorted_group = group.sort_values(by="len_ant", ascending=True)
        rows = list(sorted_group.itertuples())
        
        # Iterate through potential COMPLEX rules
        for i in range(len(rows)):
            r_complex = rows[i]
            
            # Optimization: If complex rule is already dropped, skip
            if r_complex.Index in indices_to_drop: continue

            # Check against all simpler rules
            for j in range(i):
                r_simple = rows[j]
                
                # Check subset: Is Simple a subset of Complex?
                if r_simple.ant_frozen < r_complex.ant_frozen:
                     # Check Lift: Keep complex ONLY if it improves lift significantly
                     if r_complex.lift <= (r_simple.lift * threshold_factor):
                        indices_to_drop.add(r_complex.Index)
                        break # Found one simpler rule that makes this redundant
                        
    count_removed = len(indices_to_drop)
    filtered_rules = rules.drop(index=list(indices_to_drop)).drop(columns=['ant_frozen'])
    
    return filtered_rules, count_removed

