# Code Structure Guidelines: Standardized Output Format

## Overview
All results from TSP solver processing must be centralized in the `\results` subdirectory with a consistent timestamp-based naming convention. This ensures traceability and prevents file conflicts across different runs.

## Output Directory Structure
```
\results\
├── YY_M_DD_HHMM_<instance_name>/     # Base directory for each run
│   ├── YY_M_DD_HHMM_<instance_name>.tour    # Tour solution file
│   ├── YY_M_DD_HHMM_<instance_name>.json    # Run results/metrics
│   ├── YY_M_DD_HHMM_<instance_name>.yaml    # Configuration used
│   └── seed_*.jsonl                         # Individual seed logs
└── archives/                                # Archived runs
```

## Naming Convention
**Format**: `YY_M_DD_HHMM_<instance_name>.<extension>`

**Components**:
- `YY`: 2-digit year (e.g., `25` for 2025)
- `M`: 1-2 digit month (e.g., `9` for September, `12` for December)  
- `DD`: 2-digit day (e.g., `12`)
- `HHMM`: 24-hour time format (e.g., `1805` for 6:05 PM)
- `<instance_name>`: TSPLIB instance name without `.tsp` extension (e.g., `pla33810`)
- `<extension>`: File type (`.tour`, `.json`, `.yaml`)

**Examples**:
- `25_9_12_1805_pla33810.tour` - Tour file for pla33810.tsp run on Sept 12, 2025 at 6:05 PM
- `25_12_01_0930_att48.json` - Results file for att48.tsp run on Dec 1, 2025 at 9:30 AM
- `25_9_12_1805_fnl4461.yaml` - Config file for fnl4461.tsp run

## File Types and Contents

### 1. Tour Files (.tour)
TSPLIB-compliant tour format per Update_19 specifications:
```
NAME: <instance_name>
TYPE: TOUR
DIMENSION: <n>
TOUR_SECTION
<city1>
<city2>
...
<cityN>
-1
EOF
```

### 2. Results Files (.json)
Standardized metrics and performance data:
```json
{
  "instance": "pla33810",
  "timestamp": "2025-09-12T18:05:00Z",
  "best_length": 12345,
  "wall_time_s": 3600,
  "gap_to_optimal_pct": 2.5,
  "seeds_completed": [42, 123, 456, 789, 999],
  "parity": "OK"
}
```

### 3. Configuration Files (.yaml)
Complete solver configuration used for the run:
```yaml
solver:
  candidate: {...}
  zones: {...}
  termination:
    wall_time_s: 3600
instance:
  name: "pla33810"
  path: "data/tsplib/pla33810.tsp"
run_metadata:
  timestamp: "2025-09-12T18:05:00Z"
```

## Integration with output_app.py

All processing blocks (runners, solvers, analyzers) must feed their output through `output_app.py` which will:

1. **Standardize timestamps**: Convert all timestamps to YY_M_DD_HHMM format
2. **Validate filenames**: Ensure naming convention compliance
3. **Create directories**: Set up proper `\results\` structure
4. **Handle conflicts**: Append suffixes if timestamp collisions occur
5. **Archive management**: Move older runs to archives as needed

### Required Integration Points

**All Python scripts must import and use**:
```python
from output_app import StandardOutputManager

# Initialize with instance name
output_mgr = StandardOutputManager(instance_name="pla33810")

# Get standardized paths
tour_path = output_mgr.get_tour_path()      # \results\25_9_12_1805_pla33810.tour
json_path = output_mgr.get_results_path()   # \results\25_9_12_1805_pla33810.json
yaml_path = output_mgr.get_config_path()    # \results\25_9_12_1805_pla33810.yaml
```

## Migration Requirements

### Current Non-Compliant Patterns (to be updated):
- `seed_*.jsonl` → Move to subdirectory under timestamped folder
- `*_results.json` → Rename to standardized format
- `resolved_params.yaml` → Rename to standardized format
- Various timestamp formats → Convert to YY_M_DD_HHMM

### Update Priority:
1. **High**: Main runners (`run_large3.py`, `run_auto_parallel.py`, `run_benchmarks.py`)
2. **Medium**: Analysis scripts (`analyze_cli.py`, batch runners)
3. **Low**: Archive scripts (already deprecated)

## Validation Rules

### output_app.py Must Enforce:
1. All outputs go to `\results\` subdirectory
2. Timestamp format is exactly YY_M_DD_HHMM
3. Instance name matches TSPLIB filename (without .tsp)
4. No spaces or special characters in filenames
5. Extension matches content type (.tour/.json/.yaml)

### Error Conditions:
- Invalid timestamp format → Auto-correct or raise exception
- Missing instance name → Require explicit parameter
- Path outside `\results\` → Force redirect to results directory
- File collision → Append suffix (_001, _002, etc.)

## Implementation Status
- [ ] Create `output_app.py` with `StandardOutputManager` class
- [ ] Update all runner scripts to use standardized output
- [ ] Migrate existing result files to new format
- [ ] Update documentation and examples
- [ ] Test with all TSPLIB instances

## Notes
- This format ensures chronological sorting by filename
- Human-readable timestamps for easy identification
- Consistent with TSPLIB naming conventions
- Compatible with existing analysis tools after migration
- Archive strategy prevents results directory bloat

---
*This guideline implements centralized output management for the Bee-TSP solver system with timestamped, traceable result files.*