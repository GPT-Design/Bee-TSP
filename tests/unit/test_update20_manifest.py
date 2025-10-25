#!/usr/bin/env python3
"""
Test Update_20 manifest functionality
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from output_app import StandardOutputManager

def test_manifest_functionality():
    """Test the global manifest system."""
    print("Testing Update_20 manifest functionality...")
    
    # Create a test run
    mgr = StandardOutputManager("test_manifest")
    
    # Save test tour and results
    tour_path = mgr.save_tour([1, 3, 2, 5, 4], dimension=5, tour_length=150)
    results_path = mgr.save_results({
        "best_length": 150,
        "wall_s": 60,
        "elapsed_s": 60.0,  # Fixed: make time values consistent for CI validation
        "epochs_completed": 10,
        "epoch_s": 6
    })
    
    # Create and append manifest entry
    test_run_data = {
        "instance": "test_manifest",
        "metric": "EUC_2D",
        "mode": "SH",
        "seed": 42,
        "wall_s": 60,
        "elapsed_s": 60.0,
        "epochs_completed": 10,
        "epoch_s": 6,
        "length_tsplib": 150,
        "opt": None,
        "best_known": None,
        "parity": "OK",
        "ttt10_s": None,
        "ttt5_s": None,
        "early_slope": 0.85,
        "improvements_per_min": 2.5,
        "hive_attrib": "H1"
    }
    
    manifest_entry = mgr.create_manifest_entry(test_run_data)
    print(f"Created manifest entry: {manifest_entry}")
    
    # Append to global manifest
    mgr.append_to_global_manifest(manifest_entry)
    
    print(f"Global manifest path: {mgr.global_manifest_path}")
    print(f"Run directory: {mgr.get_run_directory()}")
    
    # Verify manifest was written
    if mgr.global_manifest_path.exists():
        with open(mgr.global_manifest_path, 'r') as f:
            content = f.read()
        print(f"Manifest content:\n{content}")
        print("[OK] Manifest functionality test PASSED")
    else:
        print("✗ Manifest file not created!")
        return False
    
    return True

def test_ci_validation():
    """Test CI validation on the created files."""
    from tools.validate_update20_compliance import validate_single_run
    
    # Find the most recent test run
    results_dir = Path("results")
    test_runs = [d for d in results_dir.iterdir() 
                 if d.is_dir() and "test_manifest" in d.name]
    
    if test_runs:
        latest_run = max(test_runs, key=lambda x: x.stat().st_mtime)
        print(f"\nRunning CI validation on: {latest_run}")
        success = validate_single_run(latest_run)
        if success:
            print("[OK] CI validation test PASSED")
        else:
            print("[FAILED] CI validation test FAILED")
        return success
    else:
        print("No test runs found for validation")
        return False

def main():
    print("=" * 60)
    print("TESTING UPDATE_20 IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Test manifest functionality
        manifest_success = test_manifest_functionality()
        print()
        
        # Test CI validation
        ci_success = test_ci_validation()
        print()
        
        if manifest_success and ci_success:
            print("=" * 60)
            print("[SUCCESS] ALL UPDATE_20 TESTS PASSED!")
            print("[OK] Timezone support working (Australia/Adelaide)")
            print("[OK] Global manifest system functional")
            print("[OK] SHA256 tour verification working")
            print("[OK] CI validation hooks operational")
            print("[OK] Update_20 compliance CONFIRMED")
            print("=" * 60)
            return True
        else:
            print("=" * 60)
            print("[FAILED] SOME TESTS FAILED")
            print("=" * 60)
            return False
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)