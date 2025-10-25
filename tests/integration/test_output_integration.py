#!/usr/bin/env python3
"""
Test script to verify the standardized output system integration.
This script tests the output_app.py integration without running the full solver.
"""

import os, sys, json, time
from pathlib import Path

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from output_app import StandardOutputManager

def test_basic_functionality():
    """Test basic StandardOutputManager functionality."""
    print("Testing basic StandardOutputManager functionality...")
    
    # Test with different instance names
    test_instances = ["att48", "pla33810", "fnl4461"]
    
    for instance in test_instances:
        print(f"  Testing instance: {instance}")
        mgr = StandardOutputManager(instance)
        
        # Test path generation
        tour_path = mgr.get_tour_path()
        results_path = mgr.get_results_path()
        config_path = mgr.get_config_path()
        
        print(f"    Tour path: {tour_path}")
        print(f"    Results path: {results_path}")
        print(f"    Config path: {config_path}")
        
        # Verify naming convention
        expected_format = f"{mgr.timestamp_str}_{instance}"
        assert expected_format in str(tour_path), f"Tour path doesn't match format: {tour_path}"
        assert expected_format in str(results_path), f"Results path doesn't match format: {results_path}"
        assert expected_format in str(config_path), f"Config path doesn't match format: {config_path}"
        
        # Test file creation
        test_data = {
            "instance": instance,
            "test_run": True,
            "timestamp": time.time()
        }
        
        test_config = {
            "solver": {"test": True},
            "instance": {"name": instance}
        }
        
        # Save test files
        saved_results = mgr.save_results(test_data)
        saved_config = mgr.save_config(test_config)
        saved_tour = mgr.save_tour([1, 2, 3, 4, 5], dimension=5, tour_length=100)
        
        # Verify files exist and have correct content
        assert saved_results.exists(), f"Results file not created: {saved_results}"
        assert saved_config.exists(), f"Config file not created: {saved_config}"
        assert saved_tour.exists(), f"Tour file not created: {saved_tour}"
        
        # Test seed log
        log_path = mgr.save_seed_log(42, [
            {"event": "improve", "t": 1.0, "best": 1000},
            {"event": "summary", "best": 900, "runtime": 5.0}
        ])
        assert log_path.exists(), f"Seed log not created: {log_path}"
        
        print(f"    [OK] All files created successfully for {instance}")

def test_filename_parsing():
    """Test filename parsing functionality."""
    print("Testing filename parsing...")
    
    test_filenames = [
        "25_9_12_1805_pla33810.tour",
        "25_12_01_0930_att48.json",
        "26_1_15_2359_fnl4461.yaml"
    ]
    
    for filename in test_filenames:
        dt, instance = StandardOutputManager.parse_timestamp_from_filename(filename)
        print(f"  {filename} -> {dt}, {instance}")
        assert dt is not None, f"Failed to parse timestamp from: {filename}"
        assert instance in filename, f"Instance name not extracted correctly: {instance}"

def test_collision_handling():
    """Test filename collision handling."""
    print("Testing collision handling...")
    
    # Create multiple managers with same timestamp for same instance
    from datetime import datetime
    fixed_time = datetime(2025, 9, 12, 18, 5, 0)
    
    mgr1 = StandardOutputManager("test_collision", custom_timestamp=fixed_time)
    mgr2 = StandardOutputManager("test_collision", custom_timestamp=fixed_time)
    
    # Create first file
    path1 = mgr1.save_results({"test": "first"})
    
    # Second should get different path due to collision handling
    path2 = mgr2.save_results({"test": "second"})
    
    print(f"  First file: {path1}")
    print(f"  Second file: {path2}")
    
    assert path1 != path2, "Collision handling failed - same paths generated"
    assert path1.exists() and path2.exists(), "Both files should exist after collision handling"
    
    print("  [OK] Collision handling working correctly")

def test_legacy_migration():
    """Test legacy file migration functionality."""
    print("Testing legacy migration...")
    
    # Create a legacy-style file
    legacy_dir = Path("test_legacy")
    legacy_dir.mkdir(exist_ok=True)
    
    legacy_file = legacy_dir / "old_results.json"
    legacy_data = {"legacy": True, "best_length": 12345}
    
    with open(legacy_file, 'w') as f:
        json.dump(legacy_data, f)
    
    # Test migration by manually reading and saving with new manager
    mgr = StandardOutputManager("att48")
    
    with open(legacy_file, 'r') as f:
        legacy_data = json.load(f)
    
    # Save using new standardized format
    new_results_path = mgr.save_results(legacy_data)
    
    # Verify migration
    assert new_results_path.exists(), "Migrated file doesn't exist"
    
    with open(new_results_path, 'r') as f:
        migrated_data = json.load(f)
    
    assert migrated_data["legacy"] == True, "Legacy data not preserved"
    assert migrated_data["instance"] == "att48", "Instance name added by StandardOutputManager"
    
    # Cleanup
    legacy_file.unlink()
    legacy_dir.rmdir()
    
    print("  [OK] Legacy migration working correctly")

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING STANDARDIZED OUTPUT SYSTEM")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        print()
        
        test_filename_parsing()
        print()
        
        test_collision_handling()
        print()
        
        test_legacy_migration()
        print()
        
        print("=" * 60)
        print("[SUCCESS] ALL TESTS PASSED!")
        print("[SUCCESS] Standardized output system is working correctly")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print("[FAILED] TEST FAILED!")
        print(f"Error: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    main()