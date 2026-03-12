# weighted-fpgrowth Library — Design Document

## Goal
Extract the weighted FP-growth algorithm from `script-rule-mining` into a
standalone published Python library (`weighted-fpgrowth`). The library must
be self-contained, usable outside biology, and designed so the Rust extension
is a drop-in backend replacement with zero API changes.

---

## Repo & Package
- **Repo name**: `weighted-fpgrowth`
- **Package name (pip)**: `weighted-fpgrowth` (uses `-` as per PyPI convention)
- **Import name (code)**: `weighted_fpgrowth` (uses `_` as per Python convention)
- **Python**: >= 3.10
- **Core dependencies**: numpy, pandas
- **Optional extras**:
  - `pip install weighted-fpgrowth[validation]` — adds `statsmodels` (needed for FDR correction / permutation test)
  - `pip install weighted-fpgrowth[all]` — all extras
- **Build system**: maturin (for future Rust extension; pure-Python first)

---

## What Moves to the Library

### From `algos/weighted_fpgrowth.py`
- Everything except `run()` (which becomes the library's public entry point,
  redesigned — see API section)

### From `utils/spatial.py`
- `_precompute_geometry` + `_build_transactions_from_cache` (Gaussian decay,
  weight normalization) — this is the algorithm's Phase A, not pipeline logic
- `is_dominated` — kept but made injectable (see API section)
- `MIN_CELLS`, `DOMINANCE_THRESHOLD` constants
- `get_neighborhoods` is NOT included — caller's responsibility.
  README will document the expected input format and provide a reference
  implementation using sklearn NearestNeighbors.

### From `utils/stats.py`
- `run_permutation_test`
- `apply_fdr_correction`

### From `utils/rules.py`
- `filter_rule_by_scores_mask` (used internally in rule generation)

### Does NOT move
- `filter_redundant_rules`, `filter_rules_by_rare_cells`, `select_top_rules`
  → post-processing, pipeline responsibility
- `get_neighborhoods` — pipeline responsibility (sklearn, heavy dependency,
  no Rust performance benefit since it's already fast with NumPy/scipy)
- BAG/WINDOW/GRID neighborhood methods
- `run_association_mining.py`, `worker_task.py`, `constants.py`

---

## Public API

### Config object
All entry points accept a typed `MineConfig` dataclass instead of a raw dict.
This gives type safety, IDE autocomplete, and maps directly to Rust's
`#[derive(FromPyObject)]` struct with the same field names.

```python
from weighted_fpgrowth import MineConfig

cfg = MineConfig(
    min_support=0.01,
    min_abs_support=10,
    min_confidence=0.7,
    min_lift=1.3,
    min_leverage=0.005,
    min_conviction=1.5,
    max_negative_lift=0.7,
    max_itemset_size=4,        # NOTE: pipeline uses MAX_RULE_LENGTH — translate at boundary
    n_permutations=1000,
    random_state=42,           # for reproducible permutation test results
    radius=25.0,
    k_neighbors=30,
    bandwidth=15.0,
    min_cells_per_patch=2,     # min neighborhood size to keep transaction
    dominance_threshold=0.9,   # skip transaction if one cell type > this fraction
)
```

Pipeline translates at the call site (note: `constants.py` uses UPPER_CASE keys,
`MineConfig` uses lowercase — explicit mapping is required):
```python
cfg = MineConfig(
    min_support=config["MIN_SUPPORT"],
    min_abs_support=config.get("MIN_ABS_SUPPORT", 0),
    min_confidence=config["MIN_CONFIDENCE"],
    min_lift=config["MIN_LIFT"],
    min_leverage=config["MIN_LEVERAGE"],
    min_conviction=config["MIN_CONVICTION"],
    max_negative_lift=config["MAX_NEGATIVE_LIFT"],
    max_itemset_size=config["MAX_RULE_LENGTH"],   # key rename here
    n_permutations=config["N_PERMUTATIONS"],
    radius=config["RADIUS"],
    k_neighbors=config["K_NEIGHBORS"],
    bandwidth=config["BANDWIDTH"],
    min_cells_per_patch=config.get("MIN_CELLS_PER_PATCH", 2),
)
```

### Level 1 — Full spatial pipeline (bio and any spatial point cloud)
Caller provides pre-built neighborhoods (list of `(center_i, neighbor_idxs)` tuples).
The library handles Gaussian decay, weight normalization, FP-tree, mining, validation.

```python
rules_df, validate_fn, stats = run(
    neighborhoods,  # list of (center_i, neighbor_idxs) — caller builds with get_neighborhoods
    coords,         # (N, 2) float64 array — spatial coordinates
    cell_types,     # (N,) array of labels
    config,         # MineConfig
    validate=False,          # if True, calls validate_fn(rules_df) before returning
    is_dominated_fn=None,    # optional callable(labels) -> bool, default built-in
)
```
Returns:
- `rules_df`: DataFrame with support, confidence, lift, leverage, conviction.
  If `validate=True`, p_value and p_value_adj columns are already attached.
- `validate_fn`: callable `validate_fn(rules_df) -> rules_df` — runs permutation
  test + FDR on any subset of rules. Use when pipeline filters rules between
  mining and validation (e.g. rare-cell filter → top-N → validate).
- `stats`: dict with sizes, cell_counts, orig, kept, backend ("python" | "rust")

Usage patterns:
```python
# Simple / one-shot: validation runs on all mined rules
rules_df, _, stats = run(..., validate=True)

# Pipeline pattern: filter first, validate only survivors (current order preserved)
rules_df, validate_fn, stats = run(..., validate=False)
rules_df = filter_rare(rules_df, ...)
candidates = select_top_n(rules_df, ...)
validated = validate_fn(candidates)
```

### Level 2 — Bring your own weight matrix (non-bio / custom pipelines)
```python
rules_df, validate_fn, stats = mine(
    W,            # (N_transactions, N_items) float64 — normalized weight matrix
    item_names,   # list[str] of length N_items
    config,       # MineConfig
    validate=False,
    # validate=True requires the two args below:
    cell_types=None,   # original labels — needed to shuffle for permutation test
    geo_cache=None,    # precomputed geometry — needed to rebuild W per permutation
)
```
Note: `validate=True` on Level 2 requires `cell_types` + `geo_cache`. Without
them, call `mine(..., validate=False)` and handle validation externally.

### Level 3 — Low-level primitives (advanced / unstable API)
> ⚠️ Level 3 is not covered by API stability guarantees. Signatures may change between minor versions.
```python
from weighted_fpgrowth.tree import WeightedFPTree
from weighted_fpgrowth.mining import mine_frequent_itemsets, min_support, itemsets_to_rules
from weighted_fpgrowth.validation import build_weight_matrix, batch_check_rules
```

---

## Internal Module Structure

```
weighted_fpgrowth/
    __init__.py          # exposes run(), mine(), MineConfig (Level 1 & 2 stable API)
    geometry.py          # _precompute_geometry, _build_transactions_from_cache, is_dominated
    tree.py              # _FPNode, WeightedFPTree
    mining.py            # mine_frequent_itemsets, min_support, itemsets_to_rules, _mine
    validation.py        # build_weight_matrix, batch_check_rules, validate_rules
    stats.py             # run_permutation_test, apply_fdr_correction
    rules.py             # filter_rule_by_scores_mask
    config.py            # MineConfig dataclass, validate_config
    exceptions.py        # ConfigError, InputError, BackendError
    _api.py              # run() and mine() implementations (calls above modules)
```

---

## `is_dominated` — Injectable Design
Default built-in behaviour is preserved. `dominance_threshold` and
`min_cells_per_patch` are now fields in `MineConfig` (not hardcoded constants),
so callers can tune them without subclassing. Users can also override the entire
function:
```python
run(..., is_dominated_fn=lambda labels: False)  # disable entirely
run(..., is_dominated_fn=my_custom_fn)
```

---

## Error Model
The library raises typed exceptions with actionable messages — never bare
`ValueError` or `AssertionError` from internal code:

```python
from weighted_fpgrowth.exceptions import (
    ConfigError,    # bad or missing MineConfig field
    InputError,     # invalid neighborhoods, coords, or cell_types
    BackendError,   # Rust backend failed unexpectedly
)
```

Examples:
- `ConfigError("min_support must be in (0, 1), got 1.5")`
- `InputError("neighborhoods contains center index 9999 but coords has only 500 rows")`

---

## Pipeline Changes (script-rule-mining)

### Config key migration table

| `constants.py` key (UPPER_CASE) | `MineConfig` field (lowercase) | Notes |
|---|---|---|
| `MIN_SUPPORT` | `min_support` | |
| `MIN_ABS_SUPPORT` | `min_abs_support` | default 0 |
| `MIN_CONFIDENCE` | `min_confidence` | |
| `MIN_LIFT` | `min_lift` | |
| `MIN_LEVERAGE` | `min_leverage` | |
| `MIN_CONVICTION` | `min_conviction` | |
| `MAX_NEGATIVE_LIFT` | `max_negative_lift` | |
| `MAX_RULE_LENGTH` | `max_itemset_size` | **renamed** |
| `N_PERMUTATIONS` | `n_permutations` | |
| `RADIUS` | `radius` | |
| `K_NEIGHBORS` | `k_neighbors` | |
| `BANDWIDTH` | `bandwidth` | |
| `MIN_CELLS_PER_PATCH` | `min_cells_per_patch` | default 2 |
| *(hardcoded 0.9)* | `dominance_threshold` | **moved into config** |
| *(not present)* | `random_state` | **new** — set for reproducibility |

### `worker_task.py` call site

`get_neighborhoods` continues to be called in `worker_task.py` before the lib call,
exactly as today. Post-processing (rare filtering, redundancy, top-N selection)
stays in `worker_task.py` after the lib call.

```python
from weighted_fpgrowth import run as wfpg_run, MineConfig

cfg = MineConfig(
    min_support=config["MIN_SUPPORT"],
    min_abs_support=config.get("MIN_ABS_SUPPORT", 0),
    min_confidence=config["MIN_CONFIDENCE"],
    min_lift=config["MIN_LIFT"],
    min_leverage=config["MIN_LEVERAGE"],
    min_conviction=config["MIN_CONVICTION"],
    max_negative_lift=config["MAX_NEGATIVE_LIFT"],
    max_itemset_size=config["MAX_RULE_LENGTH"],
    n_permutations=config["N_PERMUTATIONS"],
    random_state=42,
    radius=config["RADIUS"],
    k_neighbors=config["K_NEIGHBORS"],
    bandwidth=config["BANDWIDTH"],
    min_cells_per_patch=config.get("MIN_CELLS_PER_PATCH", 2),
    dominance_threshold=0.9,
)

rules_df, validate_fn, stats = wfpg_run(
    neighborhoods, coords, cell_types, cfg, validate=False,
)
logger.info(f"Backend: {stats['backend']}")
# pipeline then: filter_rare → select_top_n → validate_fn(candidates)
```

---

## Rust Extension Design

### Philosophy
- Python (inside the lib) handles Phase A: Gaussian decay + weight normalization
  (`_precompute_geometry`, `_build_transactions_from_cache`) using NumPy — already fast
- Python builds `W` (the weight matrix) from Phase A output
- Rust receives `W` and handles Phase B+C: FP-tree construction, mining,
  rule generation, and the permutation test validation hot loop
- Zero-copy handoff: `W` is passed as a contiguous NumPy array; PyO3 reads
  it as `ndarray::Array2<f64>` with no copy
- Same public Python API — Rust is an internal backend, transparent to callers

### Rust Crate Structure
```
weighted_fpgrowth_rs/   (Rust crate, built with maturin)
    src/
        lib.rs           # PyO3 module, exposes mine_rs() and check_rules_rs()
        tree.rs          # Arena-allocated FP-tree
        mining.rs        # mine_frequent_itemsets, min_support
        rules.rs         # itemsets_to_rules
        validation.rs    # batch_check_rules
```

### Data Structure: Arena-Allocated FP-Tree
```rust
struct Node {
    item:     usize,   // index into item_names
    weight:   f64,
    parent:   usize,   // index into arena Vec
    children: HashMap<usize, usize>,
    link:     Option<usize>,  // header linked list
}
struct FPTree {
    arena:  Vec<Node>,
    header: HashMap<usize, (f64, Option<usize>)>,  // item -> (total_weight, first_node)
}
```
No `Rc<RefCell>` — all references are `usize` indices into `arena`.
Cache-friendly: tree traversal = sequential array access.

### PyO3 Rust API
```rust
#[pyfunction]
fn mine_rs(
    w: PyReadonlyArray2<f64>,
    item_names: Vec<String>,
    config: MineConfig,        // #[derive(FromPyObject)]
) -> PyResult<Vec<RuleResult>>

#[pyfunction]
fn check_rules_rs(
    w: PyReadonlyArray2<f64>,
    rules: Vec<RuleInput>,
    config: CheckConfig,
) -> PyResult<Vec<bool>>
```

### Python Backend Dispatch
Auto-fallback: Rust is used when available, Python otherwise. No config needed.
```python
# weighted_fpgrowth/_backend.py
try:
    from weighted_fpgrowth_rs import mine_rs, check_rules_rs
    _BACKEND = "rust"
except ImportError:
    _BACKEND = "python"
```
- Default: auto (Rust if installed, Python otherwise)
- Override: `run(..., backend="python")` forces Python (useful for debugging/parity checks)
- `stats["backend"]` always reports which backend was used

### Build System
- `pyproject.toml` with maturin as build backend
- Pure-Python wheel when Rust is not compiled
- Rust wheel via `maturin build --release`

### Performance Expectations
| Operation                          | Python            | Rust        | Speedup     |
|------------------------------------|-------------------|-------------|-------------|
| FP-tree construction               | slow              | arena alloc | 20–50x      |
| _mine_tree recursion               | slow              | iterative   | 30–100x     |
| _min_support recomputation loop    | slow              | SIMD-ready  | 50–200x     |
| batch_check_rules (permutation)    | NumPy             | ndarray     | 2–10x       |
| **Full permutation test (1000×)**  | hours (large FOV) | minutes     | **30–100x** |

---

## What Stays in script-rule-mining

- `constants.py` — config, algo selection, paths
- `run_association_mining.py` — multi-FOV orchestration, data loading
- `worker_task.py` — per-FOV dispatch, post-processing
- `utils/spatial.py` — BAG/WINDOW/GRID methods (fpgrowth only)
- `utils/rules.py` — filter_redundant_rules, filter_rules_by_rare_cells, select_top_rules
- `algos/fpgrowth.py` — unchanged
- `algos/weighted_fpgrowth.py` — replaced by `from weighted_fpgrowth import run`

---

## Implementation Order
1. Create new repo `weighted-fpgrowth`, set up pyproject.toml with optional extras
2. Copy and reorganise modules into library structure, add `exceptions.py`
3. Implement `MineConfig` dataclass with `random_state`, `dominance_threshold`
4. Implement Level 1 `run()` and Level 2 `mine()` with `validate` flag and `backend` in stats
5. Freeze Level 1 & 2 signatures; mark Level 3 as unstable in docstrings
6. Commit golden parity test fixtures (small fixed dataset + expected output CSV)
7. Update `script-rule-mining` to import from the library using migration table
8. Verify end-to-end parity (same results before and after extraction)
9. Add Rust crate skeleton (maturin), implement arena FP-tree
10. Implement Rust `mine_rs` and `check_rules_rs`
11. Add backend dispatch + parity tests Python vs Rust (rtol/atol to be defined)
