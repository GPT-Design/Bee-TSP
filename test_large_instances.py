#!/usr/bin/env python3
"""
Test script for large TSP instances: fnl4461, pr2392, pla33810
Uses longer time budgets appropriate for larger problem sizes.
"""

import subprocess
import sys
from pathlib import Path

def run_large_instance_test():
    """Run tests on large instances with scaled time budgets."""
    
    print("=== Large Instance TSP Test ===")
    print("Testing fnl4461 (4461 cities), pr2392 (2392 cities), pla33810 (33810 cities)")
    print("Using scaled time budgets and reduced candidate k values for efficiency")
    
    # Test configurations for different instance sizes
    configs = [
        {
            "name": "Medium Scale Test (pr2392)",
            "time_budget": 60.0,     # 1 minute 
            "candidate_k": 40,       # Higher k for better quality
            "agent_time": 5.0,       # Longer agent time
            "seeds": [42, 123],      # Reduced seeds for time
        },
        {
            "name": "Large Scale Test (fnl4461)",
            "time_budget": 120.0,    # 2 minutes
            "candidate_k": 30,       # Moderate k for balance
            "agent_time": 8.0,       # Longer agent time
            "seeds": [42],           # Single seed for time
        },
        {
            "name": "Very Large Scale Test (pla33810)",
            "time_budget": 300.0,    # 5 minutes
            "candidate_k": 20,       # Lower k for scalability
            "agent_time": 15.0,      # Much longer agent time
            "seeds": [42],           # Single seed only
        }
    ]
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"Time budget: {config['time_budget']}s, k={config['candidate_k']}")
        print(f"Agent time: {config['agent_time']}s, Seeds: {config['seeds']}")
        print('='*60)
        
        # Build command
        cmd = [
            sys.executable, "test_all_three_instances.py",
            "--time-budget", str(config['time_budget']),
            "--candidate-k", str(config['candidate_k']),
            "--agent-time", str(config['agent_time']),
            "--seeds"] + [str(s) for s in config['seeds']]
        
        try:
            # Run the test
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=config['time_budget'] + 60)
            
            if result.returncode == 0:
                print("✓ Test completed successfully")
                print("\nOutput:")
                print(result.stdout)
            else:
                print("✗ Test failed")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"✗ Test timed out after {config['time_budget'] + 60} seconds")
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print("Large instance testing complete!")
    print("Check results/ directory for detailed output files")

if __name__ == "__main__":
    run_large_instance_test()