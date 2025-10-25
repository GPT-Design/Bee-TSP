# Archived Python Files

This directory contains Python files that have been archived from the main Bee-TSP_test directory.
These files are no longer in active use but are preserved for reference.

## Directory Structure

### `/debug/` - Debug and Development Scripts
- `debug_metric_mismatch.py` - Debugging metric calculation mismatches
- `debug_oropt.py` - Debugging Or-opt operation issues

### `/tests/` - Test and Experimental Scripts  
- `test_all_three_instances.py` - Testing framework for multiple instances
- `test_backend_detection.py` - Backend system detection tests
- `test_evaluation_guards.py` - Evaluation safety mechanism tests
- `test_large_instances.py` - Large instance testing scripts
- `test_multi_hive_integration.py` - Multi-hive integration tests
- `test_optimal_comparison.py` - Optimal solution comparison tests
- `test_ttt_targets.py` - Time-to-target testing

### `/batch_runners/` - Old Batch Execution Scripts
- `execute_long_wall_runs.py` - Legacy batch execution system
- `prepare_long_wall_batch.py` - Batch preparation utilities
- `run_long_wall_batch.py` - Batch runner implementation
- `monitor_batch.py` - Batch monitoring utilities

### `/utilities/` - Utility and Check Scripts
- `check_system_readiness.py` - System readiness validation
- `tripwire_check.py` - Safety tripwire checking
- `quick_check_parity.py` - Quick parity validation
- `temp_fix_duplicate.py` - Temporary fix script (duplicate of update_19_retrofix.py)

### Root Level Archives
- `ablation_studies.py` - Ablation study framework
- `analyze_results.py` - Legacy results analysis (replaced by Update_19 tools)
- `mh_lite_runner.py` - Old MH-Lite runner (replaced by bee-tsp.py CLI)
- `real_solver_interface.py` - Legacy solver interface
- `simple_mh_lite_test.py` - Simple MH-Lite testing script

## Reason for Archival

These files were archived on 2025-09-11 as part of project cleanup because:

1. **Replaced by newer implementations**: Many files were superseded by the bee-tsp.py CLI system and Update_19 implementations
2. **Experimental/debug scripts**: Development and debugging utilities no longer needed for production
3. **Duplicated functionality**: Files that duplicated functionality available elsewhere
4. **Legacy batch systems**: Old batch execution systems replaced by the standardized CLI interface

## Current Active Files (Remaining in Root)

- `bee-tsp.py` - Main CLI interface (Update_18 implementation)
- `multi_hive_solver.py` - Core solver entry point  
- `evaluation_utils.py` - Evaluation and parity checking utilities
- `update_19_validator.py` - Update_19 validation framework
- `update_19_retrofix.py` - Update_19 retro-fix implementation
- `create_update19_manifest.py` - Manifest generation for Update_19
- `tour_export.py` - Tour file export functionality
- `verify_tsplib_tour.py` - Tour verification utilities