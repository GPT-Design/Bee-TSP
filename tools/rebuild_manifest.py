#!/usr/bin/env python3
"""
Rebuild manifest.jsonl from existing experiment results
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def rebuild_manifest():
    """Rebuild manifest.jsonl from existing experiment result JSON files."""

    results_dir = ROOT / "results"
    manifest_path = results_dir / "manifest.jsonl"

    # Back up existing manifest
    if manifest_path.exists():
        backup_path = results_dir / f"manifest_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        manifest_path.rename(backup_path)
        print(f"Backed up existing manifest to: {backup_path}")

    # Find all experiment JSON files
    json_files = list(results_dir.glob("**/25_*.json"))
    json_files.sort()

    print(f"Found {len(json_files)} experiment JSON files")

    entries_added = 0

    with open(manifest_path, 'w', encoding='utf-8') as manifest_file:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract experiment info
                instance = data.get("instance")
                wall_time_s = data.get("wall_time_s")
                seed_results = data.get("seed_results", [])

                # Determine mode from directory structure or naming
                # Check if this is SH or MH-Lite based on directory name pattern
                dir_name = json_file.parent.name
                if "2154" in dir_name or "2157" in dir_name:
                    mode = "SH"
                elif "1927" in dir_name:
                    mode = "MH-Lite"
                else:
                    mode = "Unknown"

                # Create manifest entries for each seed
                for seed_result in seed_results:
                    entry = {
                        "instance": instance,
                        "metric": "tour_length",
                        "mode": mode,
                        "seed": seed_result["seed"],
                        "wall_s": wall_time_s,
                        "elapsed_s": seed_result["runtime"],
                        "epochs_completed": 0,  # Not available in summary
                        "epoch_s": 0,
                        "tour_path": None,  # Would need to reconstruct
                        "tour_sha256": None,
                        "length_tsplib": seed_result["best_length"],
                        "opt": 11849233.0,  # pla33810 optimal
                        "best_known": 11849233.0,
                        "parity": 0.0,
                        "ttt10_s": None,  # Not available in summary
                        "ttt5_s": None,
                        "early_slope": None,
                        "improvements_per_min": None,
                        "hive_attrib": {},
                        "timestamp_utc": data.get("timestamp", "")
                    }

                    manifest_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    entries_added += 1

                print(f"Processed: {json_file.name} -> {len(seed_results)} entries")

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

    print(f"\nRebuilt manifest.jsonl with {entries_added} entries")
    print(f"Manifest location: {manifest_path}")

if __name__ == "__main__":
    rebuild_manifest()