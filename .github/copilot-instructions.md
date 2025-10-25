# Bee-TSP Solver AI Agent Instructions

## Project Overview
A bio-inspired Traveling Salesman Problem (TSP) solver using bee colony algorithms with zone-based agents, local search, and tour integration. Written in Python 3.12, targeting TSPLIB benchmark instances (50–34,000 cities).

## Architecture (Read Multiple Files to Understand)

### Core Components
- **`bee_tsp/solver.py`**: Main BeeTSPSolver - zone-based bee agents, Edge Histogram Matrix (EHM), local search (2-opt/3-opt/Or-opt)
- **`bee_tsp/multi_hive.py`**: Multi-Hive system with UCB1 allocator (experimental, disabled by default per Update_22)
- **`bee_tsp/distance.py`**: Unified Distance factory - CRITICAL for metric consistency (EUC_2D/CEIL_2D/ATT/GEO)
- **`bee_tsp/tsplib.py`**: TSPLIB parser - single source of truth for distance calculations
- **`output_app.py`**: StandardOutputManager - all results MUST flow through this (Update_20 compliance)
- **`multi_hive_solver.py`**: Main entry point bridging CLI to BeeTSPSolver
- **`bee-tsp.py`**: CLI wrapper mapping user commands to multi_hive_solver.py

### Data Flow (Critical Pattern)
1. TSPLIB file → `load_tsplib()` → metric_tag (EUC_2D/CEIL_2D/ATT/GEO)
2. Create `Distance(metric_tag, coords)` object ONCE at init
3. Pass `dist.d(i, j)` everywhere - NEVER recalculate distances ad-hoc
4. Controller re-sums tour using TSPLIB integer metric (single source of truth)
5. All outputs → `StandardOutputManager` → `results/<timestamp>_<instance>/`

## Critical Conventions (Project-Specific)

### Metric Lock (Update_24/25 - Zero Tolerance)
**NEVER** calculate distances without using the `Distance` object. The project suffered a CEIL_2D vs EUC_2D drift bug causing 417k→486k regression on pr2392.

```python
# ✅ CORRECT: Use Distance factory
from bee_tsp.distance import Distance
dist = Distance(metric_tag, coords)  # metric_tag from TSPLIB
edge_len = dist.d(i, j)  # Always use this

# ❌ WRONG: Ad-hoc distance calculation
edge_len = int(round(math.hypot(x1-x2, y1-y2)))  # Bypasses metric lock
edge_len = math.ceil(...)  # Can diverge from TSPLIB metric
```

All gain calculations, candidate graphs, POPMUSIC, HK 1-tree MUST use `dist.d(i, j)`.

### Update System (Versioned Specifications)
The project evolves via numbered Updates (see `docs/Update_*.txt`). Current state:
- **Update_20**: Standardized outputs (manifest.jsonl, timestamp directories)
- **Update_22**: Multi-Hive deprecated - SH (Single-Hive) is production mode
- **Update_24**: Rollback to U22 defaults + metric lock enforcement
- **Update_25**: Distance factory mandatory everywhere

When editing, check recent Update docs for context on "why" decisions were made.

### Output System (Update_20 - Strict Schema)
All results use `StandardOutputManager` with timestamp-based naming:

```python
# ✅ CORRECT: Use output manager
from output_app import StandardOutputManager
om = StandardOutputManager(instance_name, base_results_dir="results")
tour_path = om.write_tour(tour, best_len, metric_tag, seed, wall_s)
om.append_manifest_entry({...})  # Atomic append to results/manifest.jsonl

# ❌ WRONG: Direct file writes
with open("results/tour.txt", "w") as f:  # Bypasses standardization
```

Directory structure: `results/<YY_M_DD_HHMM>_<instance>/<base>.{tour,json,yaml,jsonl}`

### Tour-First Policy (Update_15/17)
**No tour = no improvement accepted.** Reject any result without complete tour data.

```python
# ✅ CORRECT: Validate tour before accepting
if len(tour) != n:
    return {"status": "reject", "reason": "incomplete_tour"}
controller_len = tsplib_resum(tour, dist)  # TSPLIB integer SSoT

# ❌ WRONG: Accept length without tour
return {"best_len": reported_len}  # Rejected - no tour verification
```

### Configuration (YAML-Driven)
Default config: `configs/params.yaml` (currently U24: SH defaults, HK 1-tree OFF, K=1 initializers)

```yaml
solver:
  candidate:
    k: 40                    # k-NN + Delaunay edges
    edge_cap: 10             # U22 default (U23 had issues)
    portals_per_node: 5      # Boundary discipline
  multi_hive:
    enabled: false           # U22: SH production only
  hk1tree:
    enabled: false           # U24: Rollback from U23
```

Override via CLI or custom YAML. Never hard-code parameters in solver code.

## Developer Workflows

### Running Experiments
```powershell
# Primary CLI (bee-tsp.py wrapper)
python bee-tsp.py run --instance pr2392 --wall 600 --mode SH --seeds 42,43,44 `
  --halo 0.10 --portals-per-node 5 --edge-cap 10

# Direct solver invocation (advanced)
python multi_hive_solver.py benchmarks/pr2392.tsp --config configs/params.yaml `
  --time-budget 600 --output results/test.json
```

### Testing
No pytest framework. Tests are standalone scripts with `if __name__ == "__main__"` pattern:
- Root-level: `test_output_integration.py`, `test_update20_manifest.py` (integration tests)
- `tests/` directory: Contains test data and fixtures
- `_archive/tests/`: Historical integration tests (preserved for reference)
- Diagnostic scripts: `ceil_euc_test.py`, `metric_test.py` (Update_24 metric validation)
- Run directly: `python test_output_integration.py` or `python scripts/diagnostics/metric_test.py`

### Results Analysis
```powershell
# Aggregate results from manifest
python analyze_results.ps1 -Results "results" -Algo "bstsp" -Out "results/summary.csv"

# Validate Update_20 compliance
python update_19_validator.py --manifest results/manifest.jsonl

# Verify TSPLIB tour
python verify_tsplib_tour.py --tour results/<path>.tour --instance benchmarks/pr2392.tsp
```

### Debugging Metric Issues
```python
# Dual-resum sentinel (Update_24) - catch CEIL/EUC drift
L_euc = sum(dist_euc.d(tour[i], tour[(i+1)%n]) for i in range(n))
L_ceil = sum(dist_ceil.d(tour[i], tour[(i+1)%n]) for i in range(n))
if metric_tag == 'EUC_2D' and abs(L_ceil - length_tsplib) <= 0.001:
    raise ValueError("CEIL_2D used for EUC_2D instance - METRIC DRIFT")
```

## Integration Points

### TSPLIB Evaluation Chain
```
load_tsplib() → metric_tag → Distance(metric_tag, coords) 
  → dist.d(i,j) → controller_resum(tour) → length_tsplib (integer)
```

Every module (candidate builder, local search, POPMUSIC, HK) receives `dist` object - no independent distance functions.

### Manifest System (Append-Only Ledger)
`results/manifest.jsonl` - one line per run with schema:
```json
{"instance":"pr2392", "metric":"EUC_2D", "mode":"SH", "seed":42, 
 "wall_s":600, "length_tsplib":417832, "tour_sha256":"...", 
 "parity":"OK", "timestamp_utc":"2025-09-..."}
```

Atomic writes with thread lock (see `output_app.py:_manifest_lock`). Never write manifest entries manually.

## External Dependencies
- **NumPy**: Optional (candidate graphs, feature computation)
- **SciPy**: Optional (cKDTree for k-NN, Delaunay triangulation)
- **PyYAML**: Required (config loading)
- Standard library: `math`, `random`, `time`, `json`, `hashlib`, `pathlib`

Environment setup: `sklearn-env/` virtual environment (historical name, not sklearn-specific)

## Anti-Patterns (Do Not Do)

1. **Metric bypass**: Any `math.hypot()`, `math.ceil()`, or distance calculation without `dist.d(i,j)`
2. **Direct file writes**: Anything to `results/` not through `StandardOutputManager`
3. **Length without tour**: Never accept/report tour length without validated tour array
4. **NaN/Inf in JSON**: All outputs must have valid numeric types or `null`
5. **Hard-coded params**: Use YAML configs, not magic numbers in solver code
6. **Multi-Hive assumptions**: MH is experimental/archived (Update_22) - default to SH

## Key Files for Context
- `docs/Update_24.txt`, `docs/Update_25.txt`: Metric lock rationale
- `docs/Update_20.txt`: Output system specification
- `docs/Code_Cleanup_Analysis.md`: File organization and archival decisions
- `docs/BENCHMARKS.md`: Instance test protocol
- `bee_tsp/distance.py`: Authoritative distance implementation
- `bee_tsp/update_19_utils.py`: Core timing/validation infrastructure (DO NOT ARCHIVE)
- `configs/params.yaml`: Current production parameters (U24 SH defaults)
