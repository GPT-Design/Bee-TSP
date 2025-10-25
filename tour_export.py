#!/usr/bin/env python3
"""
TSPLIB .tour file export utilities per Update_Claude_14.txt
Handles .tour file generation, SHA256 hashing, and manifest logging
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def write_tsplib_tour(tour: List[int], instance_name: str, tag: str, 
                     best_len: int, metric_tag: str, seed: int,
                     wall_s: float, commit_hash: str = "unknown",
                     output_dir: str = "results") -> str:
    """
    Write TSPLIB .tour file per Update_Claude_14.txt format.
    
    Args:
        tour: Tour as 0-based indices
        instance_name: Instance name (e.g., pr2392)  
        tag: Run tag (e.g., SH_600, MH_1800)
        best_len: TSPLIB integer tour length
        metric_tag: EUC_2D, CEIL_2D, ATT, GEO
        seed: Random seed used
        wall_s: Wall time in seconds
        commit_hash: Git commit hash
        output_dir: Base output directory
        
    Returns:
        Path to written .tour file
    """
    # Create directory structure
    tour_dir = Path(output_dir) / tag / instance_name
    tour_dir.mkdir(parents=True, exist_ok=True)
    
    # Tour filename
    tour_filename = f"{instance_name}_{tag}_champion.tour"
    tour_path = tour_dir / tour_filename
    
    # Current timestamp
    timestamp = datetime.now().isoformat()
    
    # Convert 0-based to 1-based indices
    tour_1based = [city + 1 for city in tour]
    
    # Write TSPLIB format
    with open(tour_path, 'w') as f:
        f.write(f"NAME: {instance_name}_{tag}_champion\n")
        f.write(f"TYPE: TOUR\n")
        f.write(f"DIMENSION: {len(tour)}\n")
        f.write(f"COMMENT: metric={metric_tag}, best_len={best_len}, seed={seed}, ")
        f.write(f"wall_s={wall_s:.1f}, commit={commit_hash}, date={timestamp}\n")
        f.write(f"TOUR_SECTION\n")
        
        # Write tour nodes (1-based)
        for city in tour_1based:
            f.write(f"{city}\n")
        
        f.write("-1\n")
        f.write("EOF\n")
    
    return str(tour_path)


def compute_tour_sha256(tour_path: str) -> str:
    """Compute SHA256 hash of tour file."""
    with open(tour_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def append_manifest_entry(instance_name: str, tag: str, seed_set: List[int],
                         tour_path: str, tour_sha256: str, metric_tag: str,
                         best_len: int, best_known: int, started: str, 
                         finished: str, commit_hash: str = "unknown",
                         output_dir: str = "results"):
    """Append manifest.jsonl entry per Update_Claude_14.txt."""
    manifest_dir = Path(output_dir) / tag / instance_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.jsonl"
    
    entry = {
        "instance": instance_name,
        "tag": tag, 
        "seed_set": seed_set,
        "tour_path": tour_path,
        "tour_sha256": tour_sha256,
        "metric_tag": metric_tag,
        "best_len": best_len,
        "best_known": best_known,
        "started": started,
        "finished": finished,
        "commit": commit_hash
    }
    
    with open(manifest_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def enrich_json_with_tour_info(json_data: Dict[str, Any], tour_path: str,
                              tour_sha256: str, metric_tag: str, best_len: int,
                              final_hive: Optional[str] = None) -> Dict[str, Any]:
    """Add tour information to JSON per Update_Claude_14.txt."""
    json_data.update({
        "tour_path": tour_path,
        "tour_sha256": tour_sha256, 
        "metric_tag": metric_tag,
        "best_len": best_len,
        "parity": "OK"  # Assume OK unless verification fails
    })
    
    if final_hive:
        json_data["final_hive"] = final_hive
        
    return json_data


def verify_tour_sum(tour_path: str, instance_path: str, expected_len: int) -> bool:
    """
    Verify tour sums to expected length using TSPLIB evaluation.
    Returns True if tour sums correctly.
    """
    try:
        from evaluation_utils import evaluate_tour_tsplib, create_tsplib_evaluator
        from bee_tsp.tsplib import load_tsplib
        
        # Load instance
        instance_data = load_tsplib(instance_path)
        config = {'metric': {'integer': True}, 'candidate': {'k': 28}}
        
        # Read tour from file
        tour = []
        with open(tour_path, 'r') as f:
            lines = f.readlines()
        
        tour_section = False
        for line in lines:
            line = line.strip()
            if line == 'TOUR_SECTION':
                tour_section = True
                continue
            elif line == 'EOF' or line == '-1':
                break
            
            if tour_section and line.isdigit():
                tour.append(int(line) - 1)  # Convert to 0-based
        
        # Evaluate tour
        actual_len = evaluate_tour_tsplib(tour, instance_data, config)
        return actual_len == expected_len
        
    except Exception as e:
        print(f"Tour verification error: {e}")
        return False


# Test function
if __name__ == "__main__":
    # Test with a simple tour
    test_tour = [0, 1, 2, 3, 4]
    tour_path = write_tsplib_tour(
        test_tour, "test", "TEST", 1000, "EUC_2D", 42, 60.0
    )
    
    sha256 = compute_tour_sha256(tour_path)
    print(f"Test tour written to: {tour_path}")
    print(f"SHA256: {sha256}")
    
    # Test manifest
    append_manifest_entry(
        "test", "TEST", [42], tour_path, sha256, "EUC_2D", 
        1000, 900, "2025-09-03T00:00:00", "2025-09-03T01:00:00"
    )
    print("Manifest entry added")