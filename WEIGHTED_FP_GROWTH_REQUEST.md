*** Action Plan: Implementing Distance-Decay Weighted FP-Growth for Spatial Association Mining ***

---
## ✅ Infrastructure (Completed)
The codebase has been refactored to support multiple mining algorithms:
- `utils/spatial.py` — shared spatial neighborhood logic (get_neighborhoods, is_dominated)
- `utils/rules.py` — shared post-processing (filtering, redundancy, scoring)
- `utils/stats.py` — shared permutation test skeleton (run_permutation_test with injectable build_fn/check_fn)
- `algos/fpgrowth.py` — standard binary FP-Growth isolated as a self-contained algo module
- `worker_task.py` — dispatches via `constants.ALGO` to any algo's `run()` entry point
- `algos/weighted_fpgrowth.py` — Gaussian distance-decay weighted FP-Growth (CN/KNN_R only; raises ValueError for other methods)
---

1. Problem Definition (The Need)
The current pipeline utilizes standard FP-Growth for spatial association rule mining. Standard FP-Growth relies on binary transactions (an item is either present or absent). This approach fails to capture the true biological microenvironment, specifically cellular diffusion and proximity. Currently, 1 T-cell at the edge of a search radius is treated identically to 10 T-cells tightly clustered around the center cell. We need an algorithm that accounts for both quantity and distance to accurately simulate diffusion gradients.
2. The Core Solution
Transition the core mining algorithm from binary FP-Growth to Weighted Frequent Itemset Mining (WFIM). Instead of checking for mere presence (1 or 0), the algorithm will calculate a dynamic "diffusion weight" (a float) for each cell type in a neighborhood, based on a distance-decay function and the sum of occurrences.

3. Step-by-Step Implementation & MUST-HAVE Design Directives
The following steps contain critical requirements that must be implemented in the MVP to ensure biological validity and algorithmic correctness: 

## ✅ Phase A : Spatial Preprocessing & Weight Calculation (Python)
✅ Retain spatial neighborhood logic: Keep the existing spatial search methods (e.g., KNN_R).
✅ Implement Distance-Decay Function: For a given center cell A and its neighbors of type T, calculate the weight: Weight(T) = Sum( decay_function(distance(A, T_i)) ) for all i in neighbors of type T. Use an exponential decay or inverse square law.
✅ Transaction Formatting: Convert the output to dictionaries of floats representing the diffusion weight (e.g., {'T_cell': 2.45, 'B_cell': 0.8}).
✅ MUST-HAVE: Transaction Weight Normalization
Action: Normalize the weights within each transaction so they sum to 1 (or divide by the max weight in the local window).
Why it's required: Tissues are highly heterogeneous. A very dense FOV might yield massive absolute diffusion scores (e.g., sum=20) while a sparse FOV yields tiny scores (e.g., sum=1.5). Without normalization, a single dense area will completely skew the global support threshold, creating false-positive global rules.

## ✅ Phase B: Custom Weighted FP-Tree Construction
✅ Standard libraries (like mlxtend) do not support floating-point weights. You must implement a custom tree.
✅ Node Structure: Replace the standard integer count variable with a floating-point weight variable.
✅ Header Table: Accumulate total float weights across the dataset for each item, rather than absolute counts.
✅ Tree Insertion: When inserting a transaction, increment the nodes by the specific weight of the item in that transaction.

## ✅ Phase C: Mining Logic & The Downward Closure Problem (CRITICAL)

**The original implementation used tree-propagation support** (accumulating the *suffix* item's weight into ancestor prefix paths during conditional tree construction). This was incorrect: it made `support({A,B})` depend on traversal order (asymmetric), violated downward closure, and produced lift values inconsistent with the permutation test validation.

**Correct support definition — min-based:**
`support(I) = (1/N) × Σ_t min(w_i_t for i in I)`

Co-occurrence strength is limited by the weakest partner per transaction. If B is distant (w_B=0.05) but A is central (w_A=0.8), the joint association is 0.05 — correctly reflecting that B barely participates. This definition is symmetric and biologically grounded.

**Downward closure is restored under min-based support:**
`support({A,B}) = Σ_t min(w_A, w_B) ≤ Σ_t w_A = support({A})`

So if {A} is infrequent, {A,B} is guaranteed infrequent. Standard pruning (`total_weight >= min_support`) is an exact condition — no upper-bound approximation is needed.

**GMAXW removed:**
The previous workaround (`total_weight * local_max_weight >= min_support`) was introduced specifically because tree-propagation support broke downward closure. With min-based support, downward closure holds and the header table simplifies from `[total_weight, max_weight, first_node]` to `[total_weight, first_node]`.

**Implementation note:**
After `_mine_tree` identifies candidate frequent itemsets, support is recomputed from scratch using `_min_support(itemset, transactions)`. This is necessary because the FP-tree's accumulated conditional weights (`v[0]`) are still an upper bound on min-based support, not the exact value — the tree may retain itemsets that fall below the threshold once exact support is computed.


5. ** OPTIONAL IMPROVEMENTS (For Future Iterations) ** 
The following concepts from data mining literature can optimize the algorithm further, but should be SKIPPED in the initial MVP to keep development fast and focused.
- Minimum Improvement Constraint (minimp during mining)
What it is: An algorithm constraint that drops a complex rule (e.g., A, B -> C) if its confidence isn't significantly higher than its simpler sub-rule (A -> C).
Why it helps: Prevents a combinatorial explosion of redundant rules in highly dense datasets.
Why it's OPTIONAL: Implementing this inside the FP-Tree traversal is highly complex. The current Python pipeline already handles redundancy removal excellently in Post-Processing (Step 8: Prune Redundant Rules, FDR). We can rely on the post-processor for now.

- Weight-Confidence (w-confidence / Affinity Measure)
What it is: A measure that checks if items in a rule have similar weight levels (e.g., dropping a rule where cell A has a weight of 5.0 and cell B has a weight of 0.1).
Why it helps: Filters out rules where one cell type strongly dominates the interaction while the other is negligible, finding "balanced" interactions.
Why it's OPTIONAL: In biology, asymmetric interactions are often real and important (e.g., 10 Macrophages surrounding 1 T-cell). Strict affinity pruning might discard valid biological microenvironments.

- Ordered FP-Tree (Sorting nodes by ID)
What it is: Reordering the children of nodes in the FP-Tree based on Item-ID rather than keeping them unordered.
Why it helps: Reduces the time spent searching for child nodes during tree construction and slightly compresses memory.
Why it's OPTIONAL: If implemented in Rust using a Vector-based tree, the performance will already be orders of magnitude faster than Python. The slight optimization gained by ordering IDs is not worth the added sorting complexity in the first version.

- *Architectural Consideration: The Rust Option (For Performance)*
Pure Python is highly inefficient at deep recursion and complex object traversal (which FP-Trees rely on heavily). Executing a custom pure-Python FP-Tree over millions of cells might cause severe performance bottlenecks.
Design Suggestion: Write the Weighted FP-Tree construction and mining logic in Rust, exposing it to Python via PyO3/Maturin.
Rust Caveat: Do not use standard pointers/references (Rc<RefCell<T>>) for the tree nodes, as the borrow checker makes graph traversal highly complex. Instead, use Arena Allocation or a Vector-based Graph (a flat Vec<Node> where parent/child references are usize indices). This is extremely fast and cache-friendly.
Keep the spatial distance calculations (Phase A) in Python using vectorized NumPy/SciPy, passing only the final normalized dictionaries to Rust.

6. Acceptance Criteria for MVP
✅ The system accepts transactions containing float weights representing distance-decay.
✅ Transactions are correctly normalized before tree insertion.
✅ The custom FP-Tree accumulates float weights instead of integer counts.
✅ Pruning logic uses exact standard pruning (`total_weight >= min_support`) — valid because min-based support restores downward closure. GMAXW upper-bound workaround removed.
✅ The output successfully integrates back into the existing Python post-processing pipeline (Lift calculation, FDR, redundancy pruning). [Pipeline dispatch + shared post-processing utils are ready]

2. **Technical Explanation: Distance-Decay (Diffusion) Calculation:** 
   *   **The Math:** To accurately simulate a biological diffusion gradient, the algorithm must use a Gaussian Kernel function. For a given center cell $A$, calculate the aggregated diffusion weight for each specific neighbor cell type (e.g., type $T$).
   *   **Step 1 (Individual Cell Weight):** For every single neighbor cell $i$ of type $T$ within the radius, calculate its individual weight based on its Euclidean distance $d_i$ from the center cell $A$, using the formula: 
       `w_i = exp(-0.5 * (d_i / b)^2)`
       *(Note: `b` is the bandwidth parameter, which can default to the search radius or a user-defined decay constant).*
   *   **Step 2 (Aggregation for Quantity):** Sum the individual weights of all cells of type $T$ to get the final diffusion score for that cell type in the current transaction:
       `Weight(T) = Sum(w_i)`
   *   **Why it's required:** This guarantees that 10 cells at a medium distance can cumulatively exert the same diffusion influence as 1 cell directly adjacent to the center, perfectly simulating biological proximity and quantity.
