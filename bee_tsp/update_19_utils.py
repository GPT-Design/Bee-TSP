#!/usr/bin/env python3
"""
Update_19 Utilities: Log & Metrics Integrity (Long-Wall)
Implementation of Update_19.txt requirements for timing, NaN prevention, gap metrics, etc.
"""

import time
import math
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime, timezone


class MonotonicTimer:
    """Update_19.1: Monotonic timekeeping with integrity checks."""
    
    def __init__(self):
        self.start_t = time.monotonic()
        self.wall_s = None
        self.epoch_s = None
        
    def elapsed(self) -> float:
        """Get elapsed time since start."""
        return time.monotonic() - self.start_t
    
    def finalize(self, wall_s: float, epochs_completed: int, epoch_s: float) -> Dict[str, Any]:
        """
        Update_19.1.2: Generate timing summary with integrity check.
        Returns timing data with integrity validation.
        """
        elapsed_s = self.elapsed()
        elapsed_from_epochs_s = epochs_completed * epoch_s
        
        timing_data = {
            'wall_s': wall_s,
            'elapsed_s': elapsed_s,
            'epochs_completed': epochs_completed,
            'epoch_s': epoch_s,
            'elapsed_from_epochs_s': elapsed_from_epochs_s
        }
        
        # Update_19.1.3: Sanity check (auto-fail logging)
        expected_elapsed = min(wall_s, elapsed_from_epochs_s)
        if abs(elapsed_s - expected_elapsed) / wall_s > 0.02:
            timing_data['time_integrity'] = 'FAIL'
            raise RuntimeError(f"Time integrity check failed: elapsed={elapsed_s:.2f}s, expected≈{expected_elapsed:.2f}s")
        else:
            timing_data['time_integrity'] = 'OK'
        
        # Update_19.1.4: Set final total_time_s
        timing_data['total_time_s'] = min(elapsed_s, wall_s)
        
        return timing_data


def safe_gap_metric(best_len_int: int, reference_int: Optional[int]) -> Optional[float]:
    """
    Update_19.3: Safe gap calculation with proper null handling.
    Returns gap percentage or None if reference is invalid.
    """
    if reference_int is None or reference_int <= 0:
        return None
    return 100.0 * (best_len_int - reference_int) / reference_int


def validate_no_nan_inf(data: Any, path: str = "root") -> bool:
    """
    Update_19.2.3 & 8.1: NaN/Infinity validator for JSON data.
    Recursively checks for NaN/Inf values and raises if found.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            validate_no_nan_inf(value, f"{path}.{key}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            validate_no_nan_inf(item, f"{path}[{i}]")
    elif isinstance(data, float):
        if math.isnan(data):
            raise ValueError(f"NaN found at {path}")
        if math.isinf(data):
            raise ValueError(f"Infinity found at {path}")
    
    return True


class AllocatorLogger:
    """Update_19.2: Allocator initialization without NaN values."""
    
    def __init__(self):
        self.allocations: Dict[str, int] = {}
        self.slices: Dict[str, float] = {}
        self.initialized = False
    
    def initialize_hives(self, hive_names: List[str]):
        """Initialize allocations to 0 for all hives."""
        for hive in hive_names:
            self.allocations[hive] = 0
            self.slices[hive] = 0.0
        self.initialized = True
    
    def get_safe_allocations(self) -> Dict[str, int]:
        """Return allocations dict, omitting uncomputed values."""
        if not self.initialized:
            return {}
        return self.allocations.copy()
    
    def get_safe_slices(self) -> Dict[str, float]:
        """Return slices dict, omitting uncomputed values."""
        if not self.initialized:
            return {}
        return {k: v for k, v in self.slices.items() if v > 0}


class GapMetricsCalculator:
    """Update_19.3: Well-defined gap metrics with safe calculations."""
    
    def __init__(self, instance_name: str):
        self.instance_name = instance_name
        self.best_known_int: Optional[int] = None
        self.optimal_int: Optional[int] = None
        
        # Load known values (could be from CSV or config)
        self._load_reference_values()
    
    def _load_reference_values(self):
        """Load best known and optimal values for the instance."""
        # Known values from Update_19 context and best_known.csv
        known_values = {
            'pr2392': {'best_known': 378032, 'optimal': 378032},
            'fnl4461': {'best_known': 182566, 'optimal': None},
            'pla33810': {'best_known': 66048945, 'optimal': 66048945}
        }
        
        if self.instance_name in known_values:
            values = known_values[self.instance_name]
            self.best_known_int = values.get('best_known')
            self.optimal_int = values.get('optimal')
    
    def calculate_gaps(self, best_len_int: int, dual_lb: Optional[float] = None) -> Dict[str, Optional[float]]:
        """
        Update_19.3.2: Calculate all gap metrics safely.
        Returns dict with gap_to_best_pct, gap_to_opt_pct, dual_gap_pct.
        """
        gaps = {
            'gap_to_best_pct': safe_gap_metric(best_len_int, self.best_known_int),
            'gap_to_opt_pct': safe_gap_metric(best_len_int, self.optimal_int),
            'dual_gap_pct': None
        }
        
        # Update_19.3.5: Dual gap (HK 1-tree)
        if dual_lb is not None and dual_lb > 0:
            gaps['dual_gap_pct'] = safe_gap_metric(best_len_int, int(dual_lb))
        
        # Update_19.3.4: Gap integrity validation
        for gap_type, gap_value in gaps.items():
            if gap_value is not None and gap_value > 1000:
                print(f"WARNING: {gap_type} = {gap_value:.1f}% > 1000%, inspect reference value")
        
        return gaps


class DiagnosticLogger:
    """Update_19.5: Enhanced logging schema with diagnostic fields."""
    
    def create_improvement_event(self, eval_source: str, controller_len_int: int, 
                                reported_len_raw: Optional[int], delta_ok: bool) -> Dict[str, Any]:
        """Create improvement/reject event per Update_19.5."""
        event = {
            'eval_source': eval_source,
            'controller_len_int': controller_len_int,
            'reported_len_raw': reported_len_raw,
            'delta_ok': delta_ok
        }
        validate_no_nan_inf(event)
        return event
    
    def create_epoch_diagnostics(self, time_breakdown: Dict[str, float], 
                                cross_zone_pct: float, portal_removals_pre80: int,
                                oropt_survival_pct: float, candidate_cap_violations: int,
                                hk_dual_gap_pct: Optional[float], stall_flag: bool,
                                merge_event: Optional[str]) -> Dict[str, Any]:
        """Create epoch diagnostics per Update_19.5."""
        diagnostics = {
            'time_breakdown': time_breakdown,
            'cross_zone_pct': cross_zone_pct,
            'portal_removals_pre80': portal_removals_pre80,
            'oropt_survival_pct': oropt_survival_pct,
            'candidate_cap_violations': candidate_cap_violations,
            'hk_dual_gap_pct': hk_dual_gap_pct,
            'stall_flag': stall_flag,
            'merge_event': merge_event
        }
        validate_no_nan_inf(diagnostics)
        return diagnostics


class ManifestGenerator:
    """Update_19.6: Manifest JSONL generation system."""
    
    def __init__(self, manifest_path: str = "results_manifest.jsonl"):
        self.manifest_path = Path(manifest_path)
    
    def create_manifest_entry(self, instance: str, metric: str, mode: str, seed: int,
                             timing_data: Dict[str, Any], tour_path: str, tour_sha256: str,
                             length_tsplib: int, opt: Optional[int], best_known: Optional[int],
                             parity: str, ttt10_s: Optional[int] = None, ttt5_s: Optional[int] = None,
                             early_slope: Optional[float] = None, improvements_per_min: Optional[float] = None,
                             hive_attrib: Optional[str] = None) -> Dict[str, Any]:
        """
        Update_19.6: Create manifest entry with all required fields.
        """
        entry = {
            'instance': instance,
            'metric': metric,
            'mode': mode,
            'seed': seed,
            'wall_s': timing_data['wall_s'],
            'elapsed_s': timing_data['elapsed_s'],
            'epochs_completed': timing_data['epochs_completed'],
            'epoch_s': timing_data['epoch_s'],
            'tour_path': str(Path(tour_path).resolve()),
            'tour_sha256': tour_sha256,
            'length_tsplib': length_tsplib,
            'opt': opt,
            'best_known': best_known,
            'parity': parity,
            'ttt10_s': ttt10_s,
            'ttt5_s': ttt5_s,
            'early_slope': early_slope,
            'improvements_per_min': improvements_per_min,
            'hive_attrib': hive_attrib,
            'timestamp_utc': datetime.now(timezone.utc).isoformat()
        }
        validate_no_nan_inf(entry)
        return entry
    
    def append_manifest(self, entry: Dict[str, Any]):
        """Append entry to manifest JSONL file."""
        with open(self.manifest_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=str) + '\n')


def retro_fix_timing(result_file: Path) -> Dict[str, Any]:
    """
    Update_19.9: Retro-fix procedure for existing runs.
    Recompute timing from event logs and update results.
    """
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Extract events if available
    events = data.get('events', [])
    if not events:
        print(f"No events found in {result_file}, cannot retro-fix timing")
        return data
    
    # Find first and last event timestamps
    first_event = min(events, key=lambda e: e.get('timestamp', 0))
    last_event = max(events, key=lambda e: e.get('timestamp', 0))
    
    if 'timestamp' in first_event and 'timestamp' in last_event:
        # Recompute elapsed from events
        elapsed_s = last_event['timestamp'] - first_event['timestamp']
        
        # Update timing data
        epochs_completed = data.get('epochs_completed', 0)
        epoch_s = data.get('epoch_s', 120.0)
        wall_s = data.get('wall_s', data.get('total_time_s', elapsed_s))
        
        # Apply Update_19 timing rules
        elapsed_from_epochs_s = epochs_completed * epoch_s
        total_time_s = min(elapsed_s, wall_s)
        
        data.update({
            'elapsed_s': elapsed_s,
            'elapsed_from_epochs_s': elapsed_from_epochs_s,
            'total_time_s': total_time_s,
            'audit': 'Update_19_corrected'
        })
        
        print(f"Retro-fixed timing for {result_file}: elapsed={elapsed_s:.2f}s, total={total_time_s:.2f}s")
    
    return data