#!/usr/bin/env python3
"""
Test evaluation mismatch detection per Update_Claude_15HOLD.txt
Verifies that the controller correctly aborts when bridge reports wrong lengths
"""

import sys
from typing import Dict, Any, List, Optional

# Mock solver that returns mismatched lengths
class MismatchTestSolver:
    def __init__(self, instance_name: str, instance_path: str):
        self.instance_name = instance_name
        self.instance_path = instance_path
        
    def configure(self, profile: Dict[str, Any]) -> None:
        pass
        
    def warm_start(self, champion_tour: Optional[List[int]]) -> None:
        pass
        
    def improve(self, time_budget_s: float) -> Dict[str, Any]:
        """Return deliberately mismatched length vs tour."""
        # Return a simple tour
        tour = [0, 1, 2, 3, 4, 0]  # 6 city tour (including return)
        tour_proper = tour[:-1]  # Remove return for proper format
        
        # Return WRONG length that won't match TSPLIB evaluation
        return {
            "improved": True,
            "best_len": 1000,  # This will NOT match TSPLIB evaluation of the tour
            "accepted_moves": 1,
            "time_used_s": 0.1,
            "tour_if_improved": tour_proper,
            "edges_if_improved": [(i, i+1) for i in range(5)],
            "hk_bound_local": None
        }


def test_mismatch_detection():
    """Test that controller aborts on evaluation mismatch."""
    print("=== MISMATCH SENTINEL TEST ===")
    print("Testing evaluation mismatch detection per Update_Claude_15HOLD.txt")
    
    try:
        from bee_tsp.multi_hive import MultiHiveController
        from bee_tsp.tsplib import load_tsplib
        
        # Load a small test instance  
        instance_path = "data/tsplib/pr2392.tsp"
        instance_data = load_tsplib(instance_path)
        
        # Use the actual loaded instance data (but we'll test on small tour)
        
        config = {'metric': {'integer': True}, 'candidate': {'k': 4}}
        
        # Create controller
        controller = MultiHiveController({
            'multi_hive': {'enabled': True, 'epoch_s': 60},
            'hive_profiles': {'H1': {'name': 'H1', 'delta_k': 0, 'bias': None}}
        })
        
        # Initialize with TSPLIB context
        controller.initialize_for_problem(instance_data['n'], instance_data, config)
        controller.start_time = 0.0
        
        # Create mock solver that returns mismatched lengths
        mock_solver = MismatchTestSolver("test", "test_path")
        
        print("Running epoch with mismatched evaluation...")
        
        # This should ABORT with RuntimeError due to evaluation mismatch
        try:
            controller.run_epoch(mock_solver, 1.0)
            print("TEST FAILED: Controller should have aborted on mismatch!")
            return False
            
        except RuntimeError as e:
            if "EVALUATION MISMATCH" in str(e):
                print("TEST PASSED: Controller correctly detected evaluation mismatch")
                print(f"Error message: {e}")
                return True
            else:
                print(f"TEST FAILED: Wrong error type: {e}")
                return False
                
    except ImportError as e:
        print(f"TEST FAILED: Import error: {e}")
        return False
    except Exception as e:
        print(f"TEST FAILED: Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_mismatch_detection()
    sys.exit(0 if success else 1)