#!/usr/bin/env python3
"""
Or-opt Controller with Late-only Timing and Portal Protection
Based on Update_16R.txt §4
"""

import time
from typing import Dict, Any, List, Tuple, Set
from dataclasses import dataclass


@dataclass
class OrOptConfig:
    """Configuration for Or-opt controller."""
    enabled: bool = False
    forbid_until_pct: float = 0.5  # 0-50% wall: FORBID
    intra_zone_until_pct: float = 0.8  # 50-80%: intra-zone only 
    global_after_pct: float = 0.8  # ≥80%: global allowed
    min_integer_gain: int = 3  # Minimum τ for global pass
    protect_portals: bool = True  # Protect portal edges


class OrOptController:
    """
    Controller for late-only Or-opt with portal protection (Update_16R.txt §4).
    
    Timeline:
    - 0-50% wall: FORBID Or-opt
    - 50-80%: ALLOW intra-zone only, protect portal edges  
    - ≥80%: ONE global pass, require integer gain τ ≥ 3-5
    """
    
    def __init__(self, config: Dict[str, Any]):
        oropt_config = config.get('oropt', {})
        
        self.config = OrOptConfig(
            enabled=oropt_config.get('enabled', False),
            forbid_until_pct=oropt_config.get('forbid_until_pct', 0.5),
            intra_zone_until_pct=oropt_config.get('intra_zone_until_pct', 0.8),
            global_after_pct=oropt_config.get('global_after_pct', 0.8),
            min_integer_gain=oropt_config.get('min_integer_gain', 3),
            protect_portals=oropt_config.get('protect_portals', True)
        )
        
        self.wall_time_s = 0.0
        self.start_time = 0.0
        self.global_pass_done = False
        self.protected_edges: Set[Tuple[int, int]] = set()
        
    def initialize(self, wall_time_s: float, start_time: float):
        """Initialize controller with timing parameters."""
        self.wall_time_s = wall_time_s
        self.start_time = start_time
        self.global_pass_done = False
        self.protected_edges = set()
        
    def set_protected_portals(self, portal_edges: List[Tuple[int, int]]):
        """Set edges to protect (cross-zone portals)."""
        self.protected_edges = set()
        for u, v in portal_edges:
            # Normalize edge direction
            if u > v:
                u, v = v, u
            self.protected_edges.add((u, v))
    
    def get_current_phase(self, current_time: float) -> str:
        """Get current Or-opt phase based on wall time fraction."""
        if not self.config.enabled:
            return 'disabled'
            
        time_frac = (current_time - self.start_time) / self.wall_time_s if self.wall_time_s > 0 else 0.0
        
        if time_frac < self.config.forbid_until_pct:
            return 'forbidden'
        elif time_frac < self.config.intra_zone_until_pct:
            return 'intra_zone_only'
        else:
            return 'global_allowed'
    
    def can_apply_oropt(self, current_time: float, scope: str = 'any') -> bool:
        """
        Check if Or-opt can be applied in the given scope.
        
        Args:
            current_time: Current wall time
            scope: 'intra_zone' or 'global'
        """
        phase = self.get_current_phase(current_time)
        
        if phase == 'disabled' or phase == 'forbidden':
            return False
        elif phase == 'intra_zone_only':
            return scope == 'intra_zone'
        elif phase == 'global_allowed':
            if scope == 'global' and self.global_pass_done:
                return False  # Only ONE global pass allowed
            return True
        
        return False
    
    def is_edge_protected(self, u: int, v: int) -> bool:
        """Check if edge is protected (portal edge)."""
        if not self.config.protect_portals:
            return False
            
        # Normalize edge direction
        if u > v:
            u, v = v, u
        return (u, v) in self.protected_edges
    
    def can_remove_edge(self, u: int, v: int, current_time: float, integer_gain: int = 0) -> bool:
        """
        Check if edge can be removed by Or-opt.
        
        Args:
            u, v: Edge endpoints
            current_time: Current wall time
            integer_gain: Integer gain from the Or-opt move (τ)
        """
        if self.is_edge_protected(u, v):
            time_frac = (current_time - self.start_time) / self.wall_time_s if self.wall_time_s > 0 else 0.0
            
            # Never remove protected portals unless time_frac ≥ 0.8 and τ met
            if time_frac < 0.8:
                return False
            elif time_frac >= 0.8 and integer_gain >= self.config.min_integer_gain:
                return True
            else:
                return False
        
        return True  # Non-protected edges can be removed
    
    def validate_oropt_move(self, move_data: Dict[str, Any], current_time: float) -> bool:
        """
        Validate an Or-opt move according to current phase and protection rules.
        
        Args:
            move_data: Dictionary containing move information:
                - scope: 'intra_zone' or 'global'
                - edges_removed: List of (u, v) edges being removed
                - integer_gain: Integer gain from the move (τ)
                
        Returns:
            True if move is allowed, False otherwise
        """
        scope = move_data.get('scope', 'any')
        edges_removed = move_data.get('edges_removed', [])
        integer_gain = move_data.get('integer_gain', 0)
        
        # Check if Or-opt is allowed in this scope
        if not self.can_apply_oropt(current_time, scope):
            return False
        
        # Check if any protected edges would be removed
        for u, v in edges_removed:
            if not self.can_remove_edge(u, v, current_time, integer_gain):
                return False
        
        # For global moves, require minimum integer gain
        if scope == 'global':
            if integer_gain < self.config.min_integer_gain:
                return False
        
        return True
    
    def mark_global_pass_done(self):
        """Mark that the single allowed global Or-opt pass has been completed."""
        self.global_pass_done = True
    
    def get_status_summary(self, current_time: float) -> Dict[str, Any]:
        """Get current status summary for logging."""
        phase = self.get_current_phase(current_time)
        time_frac = (current_time - self.start_time) / self.wall_time_s if self.wall_time_s > 0 else 0.0
        
        return {
            'phase': phase,
            'time_frac': time_frac,
            'global_pass_done': self.global_pass_done,
            'protected_portals': len(self.protected_edges),
            'can_intra_zone': self.can_apply_oropt(current_time, 'intra_zone'),
            'can_global': self.can_apply_oropt(current_time, 'global')
        }