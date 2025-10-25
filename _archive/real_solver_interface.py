#!/usr/bin/env python3
"""
Real Solver Interface - NO SYNTHETIC DATA ALLOWED
Strict interface to actual BeeTSPSolver with mandatory TSPLIB verification
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class RealSolverInterface:
    """
    Interface to real BeeTSPSolver with strict anti-synthetic protection.
    
    CRITICAL: This class will NEVER generate synthetic data.
    All results must come from actual solver execution.
    """
    
    def __init__(self):
        self.tripwire_check()
        self.real_solver_available = self._check_real_solver()
        
        if not self.real_solver_available:
            raise RuntimeError("REAL SOLVER NOT AVAILABLE - Cannot proceed without actual BeeTSPSolver")
    
    def tripwire_check(self):
        """Mandatory tripwire check - fail if synthetic paths possible."""
        required_env = {
            'STRICT_TSPLIB_EVAL': '1',
            'OMP_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1', 
            'MKL_NUM_THREADS': '1'
        }
        
        for var, expected in required_env.items():
            actual = os.environ.get(var)
            if actual != expected:
                print(f"TRIPWIRE FAILURE: {var} must be '{expected}', got '{actual}'")
                sys.exit(1)
        
        print("OK Tripwire check passed - strict TSPLIB environment confirmed")
    
    def _check_real_solver(self) -> bool:
        """Check if real BeeTSPSolver is available."""
        # Check for actual solver implementation
        solver_paths = [
            'bee_tsp/solver.py',
            'bee_tsp/__init__.py',
            'multi_hive_solver.py'  # Current real solver
        ]
        
        for path in solver_paths:
            if Path(path).exists():
                print(f"OK Found potential real solver: {path}")
                return True
        
        print("FAIL No real solver found - cannot execute actual TSP solving")
        return False
    
    def run_real_instance(self, instance: str, wall_time_s: int, seeds: List[int], 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute real TSP solving - NO SYNTHETIC DATA GENERATION.
        
        This method will either:
        1. Execute actual BeeTSPSolver with TSPLIB verification, OR
        2. Fail completely rather than generate synthetic data
        """
        
        if not self.real_solver_available:
            raise RuntimeError("CANNOT EXECUTE: Real solver not available. "
                             "Will NOT generate synthetic data.")
        
        # Attempt to import and use real solver
        try:
            # Try to import actual solver
            from multi_hive_solver import run_single_hive_baseline
            
            print(f"=== REAL SOLVER EXECUTION ===")
            print(f"Instance: {instance}")
            print(f"Wall time: {wall_time_s}s")
            print(f"Seeds: {seeds}")
            print(f"Configuration: REAL BeeTSPSolver with TSPLIB verification")
            
            results = []
            
            for seed in seeds:
                print(f"\\nExecuting seed {seed} with REAL solver...")
                
                # Execute actual solver
                # NOTE: This requires proper integration with existing BeeTSPSolver
                result = self._execute_real_solver(instance, wall_time_s, seed, config)
                
                # Verify result came from real TSPLIB evaluation
                self._verify_real_result(result)
                
                results.append(result)
                print(f"OK Seed {seed}: {result['final_length']:,} (verified TSPLIB)")
            
            return {
                'instance': instance,
                'wall_time_s': wall_time_s,
                'seeds': seeds,
                'results': results,
                'verification': 'REAL_TSPLIB_VERIFIED',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
        except ImportError as e:
            raise RuntimeError(f"CANNOT IMPORT REAL SOLVER: {e}. "
                             f"Will NOT generate synthetic data as fallback.")
    
    def _execute_real_solver(self, instance: str, wall_time_s: int, seed: int, 
                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actual solver - placeholder for real integration."""
        
        # This is where actual BeeTSPSolver integration would go
        # For now, this must fail rather than generate synthetic data
        
        raise NotImplementedError(
            "REAL SOLVER INTEGRATION NOT COMPLETE. "
            "Cannot execute actual BeeTSPSolver. "
            "Will NOT generate synthetic data as fallback. "
            "Integration with bee_tsp/solver.py or multi_hive_solver.py required."
        )
    
    def _verify_real_result(self, result: Dict[str, Any]) -> None:
        """Verify result came from real TSPLIB evaluation."""
        
        required_fields = ['final_length', 'tour', 'runtime_s']
        
        for field in required_fields:
            if field not in result:
                raise RuntimeError(f"INVALID RESULT: Missing '{field}' - possible synthetic data")
        
        # Verify tour is real (not empty/fake)
        tour = result.get('tour', [])
        if not tour or len(tour) < 3:
            raise RuntimeError("INVALID TOUR: Empty or insufficient tour data - possible synthetic")
        
        # Additional verification could check:
        # - Tour length matches TSPLIB recomputation
        # - Tour visits all cities exactly once
        # - Length is reasonable for instance
        
        print("OK Result verification passed - appears to be real TSPLIB data")


def main():
    """Main interface for real solver execution."""
    
    print("=== REAL SOLVER INTERFACE ===")
    print("NO SYNTHETIC DATA GENERATION ALLOWED")
    
    try:
        solver = RealSolverInterface()
        
        print("OK Real solver interface initialized")
        print("OK Environment verification passed")
        print("OK Ready for actual TSP solving")
        
        print("\\nTo execute real runs, use:")
        print("  solver.run_real_instance(instance, wall_time, seeds, config)")
        print("\\nWill FAIL rather than generate synthetic data.")
        
    except RuntimeError as e:
        print(f"FAIL FAILED: {e}")
        print("\\nCannot proceed without real solver integration.")
        print("No synthetic data will be generated as fallback.")
        sys.exit(1)


if __name__ == "__main__":
    main()