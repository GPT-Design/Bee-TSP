#!/usr/bin/env python3
"""
Test solvers for validation and integration testing.
These are NOT production solvers and should never be imported by production code.
"""

import time
import random
from typing import Dict, Any, List, Optional


class DeterministicStubSolver:
    """
    Deterministic stub solver for validation per Update_Claude_11.txt E1.
    - burns time_used_s ≈ 0.05s
    - decrements best_len by fixed amount on every call for H3 only
    - returns edges to EHM
    
    WARNING: This is TEST CODE ONLY. Never import from production.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_profile = None
        self.hive_configured = False
        
        # Deterministic state
        self.initial_length = 100000
        self.h3_improvement_per_call = 500  # H3 improves by 500 per call
        self.h3_call_count = 0
        
        # Per-hive state
        self.hive_lengths = {}
        
    def configure(self, profile: Dict[str, Any]) -> None:
        """Configure for hive personality."""
        self.current_profile = profile
        hive_name = profile.get('name', 'H1')
        
        if hive_name not in self.hive_lengths:
            self.hive_lengths[hive_name] = self.initial_length
            
        self.hive_configured = True
    
    def warm_start(self, champion_tour_or_none: Optional[List[int]]) -> None:
        """Warm start - stub implementation."""
        pass
    
    def improve(self, time_budget_s: float) -> Dict[str, Any]:
        """
        Improve current solution within time budget.
        Deterministic behavior for testing.
        """
        start_time = time.time()
        
        # Burn approximately 0.05s as specified
        target_time = 0.05
        while time.time() - start_time < target_time:
            pass  # Busy wait
        
        hive_name = self.current_profile.get('name', 'H1') if self.current_profile else 'H1'
        
        if hive_name == 'H3':
            # Only H3 improves deterministically
            self.h3_call_count += 1
            current_len = self.hive_lengths[hive_name]
            new_len = current_len - self.h3_improvement_per_call
            self.hive_lengths[hive_name] = new_len
            
            # Generate simple tour for testing
            n = 100  # Fixed size for testing
            tour_if_improved = list(range(n))
            random.shuffle(tour_if_improved)
            
            # Generate edges for EHM
            edges_if_improved = []
            for i in range(min(10, len(tour_if_improved))):
                u = tour_if_improved[i]
                v = tour_if_improved[(i + 1) % len(tour_if_improved)]
                edges_if_improved.append((u, v))
            
            return {
                "improved": True,
                "best_len": new_len,
                "accepted_moves": 1,
                "time_used_s": time.time() - start_time,
                "tour_if_improved": tour_if_improved,
                "edges_if_improved": edges_if_improved,
                "hk_bound_local": None
            }
        else:
            # Other hives don't improve (H1, H2, H4, H5)
            return {
                "improved": False,
                "best_len": self.hive_lengths.get(hive_name, self.initial_length),
                "accepted_moves": 0,
                "time_used_s": time.time() - start_time,
                "tour_if_improved": None,
                "edges_if_improved": None,
                "hk_bound_local": None
            }