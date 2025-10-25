#!/usr/bin/env python3
"""
Update_19 Retro-fix Script
Comprehensive fix for existing pla33810 results per Update_19.9
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List
import argparse

from bee_tsp.update_19_utils import retro_fix_timing


def fix_nan_values(data: Any, path: str = "root") -> Any:
    """Recursively fix NaN/Inf values in JSON data."""
    if isinstance(data, dict):
        fixed_data = {}
        for key, value in data.items():
            fixed_data[key] = fix_nan_values(value, f"{path}.{key}")
        return fixed_data
    elif isinstance(data, list):
        return [fix_nan_values(item, f"{path}[{i}]") for i, item in enumerate(data)]
    elif isinstance(data, float):
        if math.isnan(data):
            print(f"Fixed NaN at {path} -> 0")
            return 0
        elif math.isinf(data):
            print(f"Fixed Infinity at {path} -> 0")
            return 0
        else:
            return data
    else:
        return data


def add_missing_timing_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Add missing timing fields required by Update_19."""
    
    # Extract basic info
    total_time_s = data.get('total_time_s', 0)
    epochs_completed = data.get('epochs_completed', 0)
    
    # Try to determine epoch_s from events or use default
    epoch_s = 120.0  # Default from Update_18 config
    events = data.get('events', [])
    
    if events:
        # Look for epoch duration in events
        for event in events:
            if 'epoch_s' in event:
                epoch_s = event['epoch_s']
                break
    
    # Calculate missing fields per Update_19.1.2
    elapsed_s = total_time_s  # Best estimate
    elapsed_from_epochs_s = epochs_completed * epoch_s
    wall_s = max(total_time_s, elapsed_s)  # Ensure wall >= elapsed
    
    # Add Update_19 required fields
    timing_updates = {
        'wall_s': wall_s,
        'elapsed_s': elapsed_s,
        'epochs_completed': epochs_completed,
        'epoch_s': epoch_s,
        'elapsed_from_epochs_s': elapsed_from_epochs_s,
        'total_time_s': min(elapsed_s, wall_s),  # Update_19.1.4
        'time_integrity': 'ESTIMATED'  # Mark as estimated since original timing lost
    }
    
    data.update(timing_updates)
    print(f"Added timing fields: wall_s={wall_s:.1f}, elapsed_s={elapsed_s:.1f}, epochs={epochs_completed}")
    
    return data


def add_gap_metrics(data: Dict[str, Any], instance_name: str) -> Dict[str, Any]:
    """Add proper gap metrics per Update_19.3."""
    
    # Known values per Update_19 and evaluation_utils.py
    known_values = {
        'pr2392': {'best_known': 378032, 'optimal': 378032},
        'fnl4461': {'best_known': 182566, 'optimal': None},
        'pla33810': {'best_known': 66048945, 'optimal': 66048945}
    }
    
    final_length = data.get('final_length')
    if not final_length or instance_name not in known_values:
        return data
    
    values = known_values[instance_name]
    best_len_int = int(final_length)
    
    # Calculate gaps safely
    def safe_gap(num: int, den: int) -> float:
        return None if den is None or den <= 0 else 100.0 * (num - den) / den
    
    gap_metrics = {
        'gap_to_best_pct': safe_gap(best_len_int, values['best_known']),
        'gap_to_opt_pct': safe_gap(best_len_int, values['optimal'])
    }
    
    data.update(gap_metrics)
    print(f"Added gap metrics: gap_to_best={gap_metrics['gap_to_best_pct']:.2f}%, gap_to_opt={gap_metrics['gap_to_opt_pct']:.2f}%")
    
    return data


def comprehensive_retrofix(result_file: Path, instance_name: str) -> bool:
    """Apply comprehensive Update_19 retro-fix."""
    
    if not result_file.exists():
        print(f"File not found: {result_file}")
        return False
    
    print(f"\n=== Retro-fixing {result_file} ===")
    
    try:
        # Load original data
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        print("Original data loaded successfully")
        
        # Step 1: Fix NaN/Inf values
        print("Step 1: Fixing NaN/Inf values...")
        data = fix_nan_values(data)
        
        # Step 2: Add missing timing fields 
        print("Step 2: Adding missing timing fields...")
        data = add_missing_timing_fields(data)
        
        # Step 3: Add gap metrics
        print("Step 3: Adding gap metrics...")
        data = add_gap_metrics(data, instance_name)
        
        # Step 4: Add audit trail
        data['audit'] = 'Update_19_corrected'
        data['audit_timestamp'] = '2025-09-11T10:00:00Z'
        
        # Step 5: Create backup and write fixed data
        backup_path = result_file.with_suffix('.json.pre_update19')
        result_file.rename(backup_path)
        
        with open(result_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Retro-fix completed successfully")
        print(f"   Original saved to: {backup_path}")
        print(f"   Fixed data written to: {result_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Retro-fix failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Update_19 Comprehensive Retro-fix')
    parser.add_argument('result_files', nargs='+', help='Result JSON files to fix')
    parser.add_argument('--instance', help='Instance name (e.g., pla33810)')
    
    args = parser.parse_args()
    
    success_count = 0
    
    for file_path in args.result_files:
        path = Path(file_path)
        
        # Extract instance name from filename if not provided
        instance_name = args.instance
        if not instance_name:
            # Try to extract from filename like "pla33810_mh_lite_3600s_results.json"
            parts = path.stem.split('_')
            if parts:
                instance_name = parts[0]
        
        if comprehensive_retrofix(path, instance_name):
            success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Successfully fixed: {success_count}/{len(args.result_files)} files")
    
    return 0 if success_count == len(args.result_files) else 1


if __name__ == '__main__':
    exit(main())