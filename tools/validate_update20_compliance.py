#!/usr/bin/env python3
"""
Update_20 CI Validation Script
Validates that all runs comply with Update_20 requirements:
- No NaN/Infinity in JSON files
- Time plausibility checks
- Tour artifact verification
- Manifest schema compliance
- No banned synthetic strings
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output_app import (
    validate_no_nans_in_file, 
    validate_time_plausibility,
    validate_tour_artifact, 
    validate_manifest_schema,
    check_banned_synthetic_strings,
    run_full_ci_validation
)

def validate_results_directory(results_dir: Path) -> bool:
    """Validate entire results directory for Update_20 compliance."""
    print(f"Validating results directory: {results_dir}")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory does not exist: {results_dir}")
        return False
    
    all_passed = True
    validated_runs = 0
    
    # Find all run directories (format: YY_M_DD_HHMM_instance)
    for item in results_dir.iterdir():
        if item.is_dir() and '_' in item.name:
            # Skip special directories
            if item.name in ['archives', 'summary', '_archive']:
                continue
                
            # Check if it looks like a standardized run directory
            parts = item.name.split('_')
            if len(parts) >= 5:  # YY_M_DD_HHMM_instance (minimum)
                print(f"\nValidating run: {item.name}")
                if not run_full_ci_validation(item):
                    all_passed = False
                validated_runs += 1
    
    # Validate global manifest if it exists
    manifest_path = results_dir / "manifest.jsonl"
    if manifest_path.exists():
        print(f"\nValidating global manifest: {manifest_path}")
        if not validate_no_nans_in_file(manifest_path):
            all_passed = False
        else:
            # Check each manifest entry schema
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            import json
                            entry = json.loads(line)
                            if not validate_manifest_schema(entry):
                                print(f"Manifest schema validation failed at line {line_num}")
                                all_passed = False
                print(f"Global manifest validation passed")
            except Exception as e:
                print(f"Failed to validate manifest entries: {e}")
                all_passed = False
    else:
        print(f"WARNING: No global manifest found at {manifest_path}")
    
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Validated runs: {validated_runs}")
    print(f"Overall result: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'='*60}")
    
    return all_passed

def validate_single_run(run_dir: Path) -> bool:
    """Validate a single run directory."""
    if not run_dir.exists() or not run_dir.is_dir():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return False
    
    return run_full_ci_validation(run_dir)

def main():
    parser = argparse.ArgumentParser(description="Validate Update_20 compliance")
    parser.add_argument("--results-dir", default="results", help="Results directory to validate")
    parser.add_argument("--single-run", help="Validate a single run directory")
    parser.add_argument("--strict", action="store_true", help="Fail on warnings")
    
    args = parser.parse_args()
    
    if args.single_run:
        run_dir = Path(args.single_run)
        success = validate_single_run(run_dir)
    else:
        results_dir = Path(args.results_dir)
        success = validate_results_directory(results_dir)
    
    if success:
        print("\n[OK] All validation checks PASSED")
        sys.exit(0)
    else:
        print("\n[FAILED] Validation checks FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()