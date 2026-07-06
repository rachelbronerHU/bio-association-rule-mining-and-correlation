Coding Task: Spatial Information Validation Pipeline
Objective:
Develop a modular Jupyter Notebook to quantify the information content of spatial interaction rules (derived from FP-Growth) by comparing the spatial entropy of the original tissue vs. a simulated tissue generated via Maximum Entropy (MaxEnt) principles.

Data Requirements:

Load data tables and results table (extracted FP-Growth rules and frequencies).
Load data using the same conventions and paths established in notebook 06.

Workflow & Architecture:
The pipeline must be developed step-by-step, one cell at a time. Do not provide the full notebook at once. After each cell, stop, explain in simple terms what the code does, and state its role in the validation pipeline. Wait for my approval before proceeding to the next cell.

Notebook Structure (Headers & Sections):

Module 1: Spatial Entropy Engine: - Import or refactor the existing KDTree scanning logic used in notebook 06 for rule mining to compute the Co-occurrence Entropy (H 
spatial
​
 ).

Ensure the engine is adapted to calculate entropy based on the target rules found in Rules_Table consistently with the mining pipeline.

Module 2: Null Model Generator: - Implement label permutation (shuffling cell_type while keeping x, y fixed) to calculate H 
null
​
 .

Module 3: MaxEnt Simulator: - Implement Simulated Annealing to generate a synthetic tissue that satisfies the frequencies in Rules_Table.

Critical: Use localized Delta Energy calculations with a Union of affected neighborhoods (affected_centers = {idx1, idx2} ∪ neighbors_1 ∪ neighbors_2) to maintain O(K) efficiency and avoid double-counting overlaps.

Module 4: Orchestrator: - Compute the Retained Information metric: R= 
H 
null
​
 −H 
orig
​
 
H 
null
​
 −H 
rules
​
 
​
 .

Coding Standards:

Language: All code, variables, and comments MUST be in English.

Organization: - Divide the notebook into clear sections and subsections.

Helper functions must be defined in cells above the main functions that call them.

Each function must be written in a separate cell.

Clarity: Keep logic simple, clean, and readable.

Process: - 1. Write the current cell.

Explain the logic and its role in the pipeline.

Pause for user confirmation .

Mathematical Constraints:

When handling directional rules (A,B→C), ensure Center and Neighbor roles are preserved.

Use Binary FP-Growth logic for the entropy calculation (presence/absence) to ensure stable probability estimation.

Ensure distance metrics remain consistent with the original pipeline (e.g., Rank Distance vs Euclidean).