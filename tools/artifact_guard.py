#!/usr/bin/env python3
"""
Artifact Guard - Update_17.txt Appendix A
Verifies tour artifacts against manifest claims with cryptographic integrity
"""

import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

try:
    from verify_tsplib_tour import parse_tsp, parse_tour, compute_length
except ImportError:
    # Fallback if verify_tsplib_tour is not available
    print("WARNING: verify_tsplib_tour not found, using minimal verification")
    
    def parse_tsp(filepath):
        """Minimal TSP parser fallback"""
        return {"dimension": 0, "coords": []}
    
    def parse_tour(filepath, dimension):
        """Minimal tour parser fallback"""
        return list(range(dimension))
    
    def compute_length(coords, tour, metric):
        """Minimal length computation fallback"""
        return 0


def sha256(p):
    """Compute SHA256 hash of file"""
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def verify_manifest_entry(rec, tsp_root):
    """
    Verify a single manifest entry.
    
    Args:
        rec: Manifest record (JSON object)
        tsp_root: Root directory for TSP files
    
    Returns:
        tuple: (success: bool, error_message: str)
    """
    try:
        # Check required fields
        required_fields = ['instance', 'tour_path', 'tour_sha256', 'metric_tag']
        for field in required_fields:
            if field not in rec:
                return False, f"Missing required field: {field}"
        
        # Verify TSP file exists
        tsp_file = Path(tsp_root) / f"{rec['instance']}.tsp"
        if not tsp_file.exists():
            return False, f"TSP file not found: {tsp_file}"
        
        # Verify tour file exists
        tour_file = Path(rec["tour_path"])
        if not tour_file.exists():
            return False, f"Tour file not found: {tour_file}"
        
        # Verify SHA256 hash
        actual_hash = sha256(tour_file)
        expected_hash = rec.get("tour_sha256")
        if actual_hash != expected_hash:
            return False, f"SHA256 mismatch: expected={expected_hash}, actual={actual_hash}"
        
        # Verify tour length if available
        if 'best_len' in rec or 'length_tsplib' in rec:
            try:
                tsp_data = parse_tsp(str(tsp_file))
                tour_indices = parse_tour(str(tour_file), tsp_data["dimension"])
                computed_length = compute_length(tsp_data["coords"], tour_indices, rec["metric_tag"])
                
                expected_length = rec.get('best_len', rec.get('length_tsplib'))
                if expected_length and int(computed_length) != int(expected_length):
                    return False, f"Length mismatch: expected={expected_length}, computed={computed_length}"
            except Exception as e:
                return False, f"Length verification failed: {e}"
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Verification error: {e}"


def main():
    """Main artifact guard function per Update_17.txt"""
    ap = argparse.ArgumentParser(description='Artifact Guard - Verify tour artifacts')
    ap.add_argument("--manifest", nargs="+", required=True, help="Manifest files to verify")
    ap.add_argument("--tsp-root", required=True, help="Root directory for TSP files")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = ap.parse_args()
    
    print("=== ARTIFACT GUARD ===")
    print(f"Verifying {len(args.manifest)} manifest files")
    print(f"TSP root: {args.tsp_root}")
    
    total_entries = 0
    verified_entries = 0
    errors = 0
    
    for manifest_path in args.manifest:
        manifest_file = Path(manifest_path)
        if not manifest_file.exists():
            print(f"[SKIP] Manifest not found: {manifest_path}")
            continue
            
        print(f"\nVerifying manifest: {manifest_path}")
        
        try:
            manifest_content = manifest_file.read_text().strip()
            if not manifest_content:
                print("[SKIP] Empty manifest file")
                continue
                
            for line_no, line in enumerate(manifest_content.splitlines(), 1):
                if not line.strip():
                    continue
                    
                total_entries += 1
                
                try:
                    rec = json.loads(line)
                    success, message = verify_manifest_entry(rec, args.tsp_root)
                    
                    if success:
                        verified_entries += 1
                        if args.verbose:
                            print(f"[OK] {rec.get('instance', 'unknown')} - {message}")
                    else:
                        errors += 1
                        print(f"[FAIL] Line {line_no}: {message}", file=sys.stderr)
                        
                except json.JSONDecodeError as e:
                    errors += 1
                    print(f"[JSON FAIL] Line {line_no}: {e}", file=sys.stderr)
                    
        except Exception as e:
            print(f"[MANIFEST FAIL] {manifest_path}: {e}", file=sys.stderr)
            errors += 1
    
    print(f"\n=== VERIFICATION SUMMARY ===")
    print(f"Total entries: {total_entries}")
    print(f"Verified: {verified_entries}")
    print(f"Errors: {errors}")
    
    if errors > 0:
        print(f"\nFAILED: {errors} verification errors found")
        sys.exit(1)
    else:
        print("\nSUCCESS: All artifacts verified")
        sys.exit(0)


if __name__ == "__main__":
    main()