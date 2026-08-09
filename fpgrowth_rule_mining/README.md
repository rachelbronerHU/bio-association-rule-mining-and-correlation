# fpgrowth_rule_mining

Finds rules like *"where there is a CD8T cell, macrophages are nearby"* — and the
opposite, *"where there is a CD8T cell, macrophages are not"* — in spatial data.

## The idea

Every cell becomes the centre of a **patch**: itself plus the cells around it. Each
patch is one **transaction**:

```
{CD8T_CENTER: 1.0, Macrophage_NEIGHBOR: 0.8, Plasma_NEIGHBOR: 0.2}
```

How much a neighbour counts is the only choice that changes the maths:

| `weighting` | a neighbour counts |
|---|---|
| `WEIGHTED` | `exp(-0.5 * (distance / bandwidth) ** 2)` — a far neighbour counts less |
| `BINARY` | `1.0` — near or far |

Support is **min-based** — a pattern is only as strong as its weakest member:

```
support(I) = mean over transactions of min(weight of each item in I)
```

With binary weights that is the plain fraction of transactions holding every item.

A rule always reads *centre → neighbours*. The centre is on the left, never the right.

## Pipeline

| step | file | what it does |
|---|---|---|
| 1 | `transactions.py` | group cells into patches — everything within `radius`, or the `k_neighbors` nearest |
| 2 | `transactions.py` | weigh each neighbour: distance decay, or a flat 1.0 |
| 3 | `transactions.py` | patch → transaction. Same-label neighbours add up, capped at 1.0. Patches dominated by one label are dropped |
| 4 | `attraction.py`, `avoidance.py` | two searches, one per claim — see below |
| 5 | `rules.py` | measure and judge. Rules naming a too-rare cell type are dropped |
| 6 | `validation/significance.py` | shuffle the labels, see how often the rule still passes → `p_value` |
| 7 | `rules.py` | drop rules a shorter rule already said (`filter_rules`) |
| 8 | `validation/false_discovery.py` | does the rule hold across the whole dataset? |

Steps 1–7 run per sample. Step 8 is the only one that looks across samples.

## Usage

**Many samples**, in the usual order:

```python
from fpgrowth_rule_mining import Settings, Weighting, Method, run_samples

settings = Settings(weighting=Weighting.WEIGHTED, method=Method.CN,
                    radius=25.0, min_support=0.01,
                    min_lift=1.2, max_items_per_rule=4,
                    avoidance_max_lift=0.8)

if __name__ == "__main__":                       # required when workers is set
    report = run_samples(
        samples,                                 # (sample_id, coords, labels) triples
        settings,
        n_shuffles=1000, random_seed=42,
        min_lift_gain=1.1,
        workers=8, output_path="results/",
    )

report.rules()                        # final rules, with a sample_id column
report.tested()                       # everything mined, raw p-values, nothing removed
report.failures                       # (sample_id, traceback) for samples that raised
report.dataset_significance(groups)   # does each rule hold across the dataset?
```

A sample that raises is recorded and the rest carry on. If every sample fails, that
raises.

**One sample**, or a different order:

```python
from fpgrowth_rule_mining import mine, filter_rules

result = mine(coords, labels, settings)          # coords (n, 2), labels (n,)
tested = result.add_p_values(n_shuffles=1000, random_seed=42)
rules  = filter_rules(tested, min_lift_gain=1.1)
```

The library writes nothing except `run_config.json`, and only if you pass
`output_path`. Results come back as DataFrames.

## Parameters

### Settings

Fields without a default must be chosen. Everything else defaults to `None` — that
threshold is not applied, so a rule is only dropped for a reason you asked for.

| | required | what it is |
|---|---|---|
| `weighting` | **yes** | `WEIGHTED` or `BINARY` |
| `method` | **yes** | `CN`: everything inside the radius. `KNN_R`: the k nearest, capped by it |
| `radius` | **yes** | how far a patch reaches |
| `min_support` | **yes** | how often a pattern must appear. `0` never terminates |
| `min_lift` | **yes** | what counts as attraction. Must be `>= 1` |
| `max_items_per_rule` | **yes** | longest rule to build, 2 to 5 |
| `avoidance_max_lift` | **when avoidance is on** | what counts as avoidance. Must be `< 1` |
| `bandwidth` | | distance at which a neighbour counts ~0.6. Unset, it follows `radius` |
| `k_neighbors` | for `KNN_R` | how many neighbours to take |
| `min_cells_per_patch` | | skip patches smaller than this. Minimum 2 |
| `max_one_type_share` | | skip a patch this dominated by one label |
| `min_patches` | | how many patches must back a rule. Counted in weight, so exact under `BINARY` and conservative under `WEIGHTED` |
| `strong_confidence` + `min_support_when_strong` | | above that confidence, allow this lower support |
| `min_confidence`, `min_leverage`, `min_conviction` | | tighten attraction further |
| `include_avoidance_rules` | | search for cell types that keep apart too. On by default |
| `avoidance_max_leverage` | | tighten avoidance further |
| `avoidance_min_expected_meetings` | | meetings that had to be expected before a miss counts. Default 10 |
| `min_label_count`, `min_label_share` | | ignore rules naming a label this rare in the sample |

`min_lift` and `avoidance_max_lift` are required because they are what makes a p-value
mean anything — see [below](#why-the-two-lift-thresholds-are-required).

### add_p_values

| | what it is |
|---|---|
| `n_shuffles` | **required.** The smallest possible p-value is `1/(n_shuffles+1)`, so 5 shuffles can never reach 0.05 |
| `random_seed` | fix it and re-runs give identical p-values |
| `labels_kept_fixed` | labels that never move. `"Name"` is exact; `"Name*"` matches anything starting with Name, so `"CD4*"` also catches `CD45` |

Two things it will not fudge:

- pin so much with `labels_kept_fixed` that nothing is left to shuffle → **raises**,
  rather than returning 1.0 everywhere and looking like a real negative
- a rule naming a cell type this sample does not have → `p_value = 1.0`. It was never
  tested, and "never survived" is not the same as "best result in the run"

### run_samples

| | what it is |
|---|---|
| `samples` | iterable of `(sample_id, coords, labels)`. Not a DataFrame, so no column names are assumed |
| `workers` | `None` runs here. An integer runs that many **processes** — mining is CPU-bound. On Windows, guard the caller with `if __name__ == "__main__"` |
| `output_path` | where to write `run_config.json`. `None` writes nothing |
| `min_lift_gain` | how much better a longer rule must be to earn its place |

Each sample derives its own seed from `random_seed`, so a parallel run matches a serial
one.

## Attraction and avoidance

*Sit together* and *keep apart* are opposite claims, so they are two searches. Every
rule records which one found it in its `kind` column (`"attracts"` or `"avoids"`).
`lift >= 1` and `lift < 1` are structural, one per search, so no rule can be both.

**Attraction** — `attraction.py`. FP-growth over the transactions. Pruning on support is
exact: adding an item never raises min-based support, so nothing pruned could come back.

- clears `min_support` (or `min_support_when_strong` when confidence is high) and
  `min_patches`
- then `min_lift`, plus `min_leverage` / `min_conviction` / `min_confidence` if set

**Avoidance** — `avoidance.py`. Cannot be the same search: low joint support *is* the
finding, and never meeting — the strongest case — has no support to prune on. No tree,
no pruning.

- **no joint-support requirement**
- enough patches hold the antecedent to measure a rate on (`min_patches`)
- enough meetings were expected that seeing none is surprising
  (`avoidance_min_expected_meetings`)
- then `avoidance_max_lift`, plus `avoidance_max_leverage` if set

### Why expected meetings, not a support bar

lift is observed over expected, so it needs enough expected to divide by. If none were
seen, `e^-expected` is the best p-value the evidence could support — expect 2 and that
is about 1 in 7; expect 20 and it is 1 in 500 million.

A support bar cannot say that, and it is not even one bar. A patch holds **one** centre
and **many** neighbours, so:

```
support as a NEIGHBOR  <=  k x support as a CENTER      k = mean neighbours per patch
```

One fraction is up to `k` times harsher on the left than the right. Past a point, no
rare cell type can be the centre of an avoidance rule at all.

**How the two checks relate.** `con_support <= 1` always, so:

```
expected meetings = ant_support x con_support x n  <=  ant_support x n = the patch count
```

The expected-meetings check therefore covers the patch check unless
`min_patches > avoidance_min_expected_meetings`. Each still catches what the other
cannot: few patches with a very common neighbour has the meetings but no rate worth
measuring; many patches with a very rare neighbour has the rate but nothing to deplete.

**What it does not do.** The bar makes *total absence* meaningful. It cannot detect a
*partial* shortfall — `lift <= 0.8` is a 20% deficit, needing an expected count near 100
to clear Poisson noise. That is what the shuffle test and the FDR correction are for.

### What bounds the avoidance search

`max_items_per_rule`, not support. But rule length alone is not enough — the count grows
with the number of items too: 20 items at length 6 is 54,000 combinations, 40 items is
4.6 million. So:

- `max_items_per_rule` is capped at 5
- above `MOST_COMBINATIONS`, the search refuses to start rather than running for days
- only combinations with **at most one centre** are measured, since a candidate set
  holds exactly one and splitting never moves it right. That is 3× fewer at length 4,
  9× at length 6

## Why the two lift thresholds are required

The shuffle test asks how often a rule still passes **its own thresholds** in a shuffled
tissue. With no lift threshold there is nothing left to fail:

- avoidance reduces to `lift < 1`, which a shuffle clears about half the time, so every
  p-value lands near 0.5 and the test carries no information
- attraction still has the support policy, so it degrades less sharply, but the same way

Every other threshold is genuinely optional. Unset means no extra tightening, not no
filtering.

## Testing many rules at once

Test 12,000 rules at 5% and about 600 look significant by luck. **The right correction
depends on the claim you are making**, which the library cannot know — so it stores raw
p-values and corrects nothing. You correct at the point of the claim.

**A claim about one sample** — *"in FOV 17, CD8T avoids Paneth."* The family is the
rules tested in FOV 17:

```python
from fpgrowth_rule_mining.validation.false_discovery import false_discovery_rates
one = report.tested().query("sample_id == 'FOV_17'")
one["sample_fdr"] = false_discovery_rates(one["p_value"])
```

**A claim about the study** — *"CD8T avoids Paneth, as a recurring feature."* Its own
question. Filtering each sample at 5% and counting survivors is invalid: each sample
gets its own 5% of false rules, so across 250 samples a pure-noise rule shows up in
about 12.

```python
groups = {sample_id: patient_id, ...}          # or None: every sample its own group
report.dataset_significance(groups=groups, alpha=0.05)
```

Four steps:

1. **One answer per group.** A patient with 8 FOVs must not get 8 votes: take the
   group's best p-value and charge it for the attempts, `min(1, m × p_best)`.
2. **Count real attempts.** `groups_tested` counts only groups where the rule *could*
   have been found. Found in 15 of 30 samples is overwhelming; the same against 250 is
   noise. A group that could have found it and didn't scores 1.0 — a failed attempt,
   not a missing one.
3. **One binomial per rule** — is `groups_passed` of `groups_tested` more than `alpha`
   predicts?
4. **Adjust once** across distinct rules → `dataset_fdr`.

Notes:

- **Which rules are asked about** — only those that survived the redundancy filter
  somewhere. A rule a shorter one already said *everywhere* is not a separate claim.
- **Which samples are counted** — all of them. Redundancy is decided per sample, so
  counting only survivors would turn the others into failures they never were.
- **It errs low.** The p-value is floored at `1/(n_shuffles+1)` and the ×m penalty
  assumes nothing about independence. `n_shuffles` must be large enough that
  `m × 1/(n_shuffles+1)` can get under `alpha`, or nothing can pass.
- **Rules are matched exactly**, roles included, so `CD8T_CENTER → Paneth_NEIGHBOR` and
  `Paneth_CENTER → CD8T_NEIGHBOR` stay separate.

## Complex rules classification

Rules with 3 or more items are classified to see if they add new information beyond their simpler pairwise parts. The library adds three columns: `rule_type`, `complex_class`, and `simpler_rules`.

Classification is based on `min_lift_gain`:

**Type I (A + B → C)**
- `"new"`: Neither `A → C` nor `B → C` were found.
- `"improved"`: The rule beats the best simpler rule's lift by `min_lift_gain`.
- `"redundant"`: Fails to improve upon the simpler rules.

**Type II (A → B + C)**
- `"consequent-driven"`: `B` and `C` strongly predict each other, forming a natural niche.
- `"new"`, `"improved"`, or `"redundant"`: Compared against `A → B` and `A → C`.

Rules are not dropped; they are labeled. The `simpler_rules` column shows exactly which rules were used for the comparison.

Reference: [Bayardo et al., *Constraint-Based Rule Mining in Large, Dense Databases*](https://www.bayardo.org/ps/icde99.pdf)
