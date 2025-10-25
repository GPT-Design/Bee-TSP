# Code Cleanup Analysis - October 2025

## Executive Summary

This analysis addresses three questions about potential code cleanup and archival:

1. **solver_backup.py**: NOT identical to solver.py - contains outdated code with different logic
2. **update_19_utils.py**: ACTIVELY USED - cannot be archived (4 active imports)
3. **Root-level scripts**: Mixed usage - some active utilities, some one-off tests

## Question 1: Is solver_backup.py identical to solver.py?

### Answer: NO - They are different files

**Evidence:**
- Binary comparison shows ~376 byte differences starting at offset 0x76EA
- `solver_backup.py` is 789 lines (same as solver.py but different content)
- Key difference at lines 470-500: solver_backup.py has outdated 3-opt logic with simplified comments

**Recommendation: ARCHIVE solver_backup.py**
- No imports found in codebase (0 references)
- Appears to be a pre-Update_24 snapshot before metric lock refactoring
- Should move to `_archive/bee_tsp/solver_backup.py` with timestamp note

**Action:**
```powershell
# Create archive location
New-Item -ItemType Directory -Path "_archive/bee_tsp" -Force

# Move with context note
Move-Item "bee_tsp/solver_backup.py" "_archive/bee_tsp/solver_backup.py"

# Add README explaining archival
@"
# solver_backup.py (Archived Oct 2025)

Pre-Update_24 backup of solver.py before metric lock enforcement.
Contains outdated 3-opt implementation and different improvement tracking logic.
Preserved for historical reference only - DO NOT USE.

See docs/Update_24.txt and docs/Update_25.txt for current implementation.
"@ | Out-File "_archive/bee_tsp/README_solver_backup.txt"
```

---

## Question 2: Can update_19_utils.py be archived?

### Answer: NO - Actively used by production code

**Active Imports Found (4 files):**

1. **update_19_validator.py** (production validator)
   ```python
   from bee_tsp.update_19_utils import validate_no_nan_inf, retro_fix_timing
   ```

2. **update_19_retrofix.py** (production retrofix tool)
   ```python
   from bee_tsp.update_19_utils import retro_fix_timing
   ```

3. **create_update19_manifest.py** (manifest generator)
   ```python
   from bee_tsp.update_19_utils import ManifestGenerator
   ```

4. **_archive/temp_fix_duplicate.py** (archived, can ignore)

**Key Functions Provided:**
- `MonotonicTimer`: Timing integrity checks (Update_19.1)
- `validate_no_nan_inf()`: NaN/Inf prevention (Update_19.2)
- `retro_fix_timing()`: Retroactive timing fixes
- `ManifestGenerator`: Manifest creation utilities
- `compute_gap_pct()`: Gap percentage calculations
- `validate_events_stream()`: Event stream validation

**Recommendation: KEEP in active code**
- Essential for Update_20 compliance validation
- Required by manifest generation pipeline
- Used by retrofix/validation tools in production workflows

**No action needed** - this module is core infrastructure.

---

## Question 3: Which root-level .py files should move to scripts/?

### Analysis of Root-Level Python Files

#### ✅ KEEP IN ROOT (Core Entry Points / Utilities)

| File | Reason | Active Imports |
|------|--------|----------------|
| `bee-tsp.py` | Primary CLI entry point | 0 (runs standalone) |
| `multi_hive_solver.py` | Core solver entry point | 6 (from scripts/, tools/, _archive/) |
| `output_app.py` | Critical output infrastructure | 8 (scripts, tools, tests) |
| `evaluation_utils.py` | Core evaluation library | 4 (multi_hive_solver, tour_export, _archive) |
| `tour_export.py` | Core export library | 3 (multi_hive_solver, _archive) |
| `verify_tsplib_tour.py` | Production verification tool | 1 (tools/artifact_guard.py) |
| `update_19_validator.py` | Production validator (CLI) | 0 (runs standalone) |
| `update_19_retrofix.py` | Production retrofix tool (CLI) | 0 (runs standalone) |
| `create_update19_manifest.py` | Manifest generation (CLI) | 0 (runs standalone) |

#### 🔄 MOVE TO scripts/ (One-Off Scripts)

| File | Reason | Recommendation |
|------|--------|----------------|
| `ceil_euc_test.py` | One-off Update_24R metric test | Move to `scripts/tests/` |
| `metric_test.py` | One-off diagnostic for pr2392 | Move to `scripts/tests/` |
| `test_output_integration.py` | Integration test script | Move to `tests/integration/` |
| `test_update20_manifest.py` | Unit test for manifest | Move to `tests/unit/` |

#### 📊 KEEP IN ROOT (For Now - PowerShell Dependencies)

| File | Reason | Note |
|------|--------|------|
| `analyze_results.ps1` | PowerShell wrapper | Keep with .bat file |
| `analyze_results.bat` | Batch wrapper | Part of analysis workflow |

---

## Detailed Recommendations

### High Priority: Archive solver_backup.py

```powershell
# Execute archival
New-Item -ItemType Directory -Path "_archive/bee_tsp" -Force
Move-Item "bee_tsp/solver_backup.py" "_archive/bee_tsp/"
```

**Rationale:** No references, outdated logic, serves only as historical snapshot.

---

### Medium Priority: Reorganize Test Files

```powershell
# Create test structure
New-Item -ItemType Directory -Path "tests/integration" -Force
New-Item -ItemType Directory -Path "tests/unit" -Force
New-Item -ItemType Directory -Path "scripts/diagnostics" -Force

# Move test files
Move-Item "test_output_integration.py" "tests/integration/"
Move-Item "test_update20_manifest.py" "tests/unit/"

# Move diagnostic scripts
Move-Item "ceil_euc_test.py" "scripts/diagnostics/"
Move-Item "metric_test.py" "scripts/diagnostics/"
```

**Rationale:** 
- Aligns with standard Python project structure
- Separates tests from core entry points
- Diagnostic scripts grouped with other scripts

**Impact:** Low - these are standalone scripts with no imports

---

### Low Priority: Update Documentation

After moves, update:

1. **README.md** (if exists) - reflect new test locations
2. **docs/Update_24.txt** - note solver_backup.py archival
3. **.github/copilot-instructions.md** - update test file locations

---

## Summary Table: Root-Level File Disposition

| Category | Count | Files | Action |
|----------|-------|-------|--------|
| **Core Entry Points** | 3 | bee-tsp.py, multi_hive_solver.py, output_app.py | **KEEP** |
| **Core Libraries** | 3 | evaluation_utils.py, tour_export.py, verify_tsplib_tour.py | **KEEP** |
| **Validators/Tools** | 3 | update_19_validator.py, update_19_retrofix.py, create_update19_manifest.py | **KEEP** |
| **Analysis Wrappers** | 2 | analyze_results.ps1, analyze_results.bat | **KEEP** |
| **Test Scripts** | 2 | test_output_integration.py, test_update20_manifest.py | **MOVE to tests/** |
| **Diagnostic Scripts** | 2 | ceil_euc_test.py, metric_test.py | **MOVE to scripts/diagnostics/** |
| **Data Files** | 2 | best_known.csv, test_instances.txt | **KEEP** (data) |
| **Workspace Config** | 1 | Bee-TSP_test.code-workspace | **KEEP** |

---

## Implementation Priority

### Phase 1: Immediate (No Breaking Changes)
1. ✅ Archive `bee_tsp/solver_backup.py` → `_archive/bee_tsp/`
2. ✅ Add archival documentation

### Phase 2: Low-Risk Reorganization
1. Move test files to `tests/` subdirectories
2. Move diagnostic scripts to `scripts/diagnostics/`
3. Update import statements if needed (minimal - these are standalone)

### Phase 3: Documentation
1. Update `.github/copilot-instructions.md` with new test locations
2. Update any relevant docs/ files

---

## Verification Commands

After implementing changes, verify:

```powershell
# Verify no broken imports
python -m py_compile bee_tsp/*.py
python -m py_compile *.py

# Run tests to ensure nothing broke
python tests/integration/test_output_integration.py
python tests/unit/test_update20_manifest.py

# Verify solver still works
python bee-tsp.py run --instance pr2392 --wall 60 --mode SH --seeds 42 `
  --halo 0.10 --portals-per-node 5 --edge-cap 10
```

---

## Notes

- **update_19_utils.py** is CORE INFRASTRUCTURE - never archive
- Most root-level .py files are legitimately CLI entry points
- The project follows a "flat root for executables" pattern which is valid
- Test file moves are optional cleanup, not critical

**Date:** October 4, 2025  
**Status:** Analysis Complete, Awaiting Approval for Implementation
