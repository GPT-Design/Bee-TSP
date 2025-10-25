#!/usr/bin/env python3
"""
Update_19 Validator: CI + tools for integrity checking
Implementation of Update_19.8 validation requirements.
"""

import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib

from bee_tsp.update_19_utils import validate_no_nan_inf, retro_fix_timing


class Update19Validator:
    """Comprehensive validator for Update_19 requirements."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def log(self, message: str, level: str = 'info'):
        """Log validation messages."""
        if level == 'error':
            self.errors.append(message)
            if self.verbose:
                print(f"ERROR: {message}")
        elif level == 'warning':
            self.warnings.append(message)
            if self.verbose:
                print(f"WARNING: {message}")
        elif self.verbose:
            print(f"INFO: {message}")
    
    def validate_no_nan_inf(self, data: Dict[str, Any], filename: str) -> bool:
        """Update_19.8.1: No-NaN validator."""
        try:
            validate_no_nan_inf(data, f"file:{filename}")
            self.log(f"NaN/Inf check passed for {filename}")
            return True
        except ValueError as e:
            self.log(f"NaN/Inf check failed for {filename}: {e}", 'error')
            return False
    
    def validate_time_integrity(self, data: Dict[str, Any], filename: str) -> bool:
        """Update_19.8.2: Time plausibility check."""
        wall_s = data.get('wall_s')
        elapsed_s = data.get('elapsed_s')
        epochs_completed = data.get('epochs_completed', 0)
        epoch_s = data.get('epoch_s', 120.0)
        
        if not all([wall_s, elapsed_s]):
            self.log(f"Missing timing data in {filename}", 'error')
            return False
        
        elapsed_from_epochs_s = epochs_completed * epoch_s
        expected_elapsed = min(wall_s, elapsed_from_epochs_s)
        
        if abs(elapsed_s - expected_elapsed) / wall_s > 0.02:
            self.log(f"Time integrity failed for {filename}: elapsed={elapsed_s:.2f}s, expected≈{expected_elapsed:.2f}s", 'error')
            return False
        
        self.log(f"Time integrity check passed for {filename}")
        return True
    
    def validate_gap_metrics(self, data: Dict[str, Any], filename: str) -> bool:
        """Update_19.8.3: Gap plausibility check."""
        gaps_valid = True
        
        for gap_key in ['gap_to_best_pct', 'gap_to_opt_pct', 'dual_gap_pct']:
            gap_value = data.get(gap_key)
            if gap_value is not None:
                # Check for valid numeric value
                if not isinstance(gap_value, (int, float)) or math.isnan(gap_value) or math.isinf(gap_value):
                    self.log(f"Invalid {gap_key} in {filename}: {gap_value}", 'error')
                    gaps_valid = False
                # Check for plausible range
                elif gap_value > 1000:
                    self.log(f"Suspicious {gap_key} in {filename}: {gap_value:.1f}% > 1000%", 'warning')
        
        if gaps_valid:
            self.log(f"Gap metrics check passed for {filename}")
        
        return gaps_valid
    
    def validate_tour_artifact(self, tour_path: str, expected_sha256: str, expected_length: int) -> bool:
        """Update_19.8.4: Artifact guard (post-run) - verify tour SHA and length."""
        if not Path(tour_path).exists():
            self.log(f"Tour file not found: {tour_path}", 'error')
            return False
        
        try:
            # Verify SHA256
            with open(tour_path, 'rb') as f:
                actual_sha256 = hashlib.sha256(f.read()).hexdigest()
            
            if actual_sha256 != expected_sha256:
                self.log(f"SHA256 mismatch for {tour_path}: expected {expected_sha256}, got {actual_sha256}", 'error')
                return False
            
            # TODO: Re-sum tour length (requires TSPLIB evaluation)
            # For now, just verify the SHA matches
            self.log(f"Tour artifact check passed for {tour_path}")
            return True
            
        except Exception as e:
            self.log(f"Tour artifact validation failed for {tour_path}: {e}", 'error')
            return False
    
    def validate_manifest_schema(self, entry: Dict[str, Any]) -> bool:
        """Update_19.8.5: Schema check - all manifest fields present with correct types."""
        required_fields = {
            'instance': str,
            'metric': str,
            'mode': str,
            'seed': int,
            'wall_s': (int, float),
            'elapsed_s': (int, float),
            'epochs_completed': int,
            'epoch_s': (int, float),
            'tour_path': str,
            'tour_sha256': str,
            'length_tsplib': int,
            'opt': (int, type(None)),
            'best_known': (int, type(None)),
            'parity': str,
            'timestamp_utc': str
        }
        
        schema_valid = True
        
        for field, expected_type in required_fields.items():
            if field not in entry:
                self.log(f"Missing required field: {field}", 'error')
                schema_valid = False
            elif not isinstance(entry[field], expected_type):
                self.log(f"Wrong type for {field}: expected {expected_type}, got {type(entry[field])}", 'error')
                schema_valid = False
        
        # Optional fields with type checking
        optional_fields = {
            'ttt10_s': (int, type(None)),
            'ttt5_s': (int, type(None)),
            'early_slope': (float, type(None)),
            'improvements_per_min': (float, type(None)),
            'hive_attrib': (str, type(None))
        }
        
        for field, expected_type in optional_fields.items():
            if field in entry and not isinstance(entry[field], expected_type):
                self.log(f"Wrong type for optional {field}: expected {expected_type}, got {type(entry[field])}", 'error')
                schema_valid = False
        
        if schema_valid:
            self.log("Manifest schema check passed")
        
        return schema_valid
    
    def validate_result_file(self, result_file: Path) -> bool:
        """Validate a complete result file."""
        if not result_file.exists():
            self.log(f"Result file not found: {result_file}", 'error')
            return False
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.log(f"Failed to parse JSON from {result_file}: {e}", 'error')
            return False
        
        all_valid = True
        
        # Run all validation checks
        all_valid &= self.validate_no_nan_inf(data, str(result_file))
        all_valid &= self.validate_time_integrity(data, str(result_file))
        all_valid &= self.validate_gap_metrics(data, str(result_file))
        
        return all_valid
    
    def validate_manifest_file(self, manifest_file: Path) -> bool:
        """Validate a manifest JSONL file."""
        if not manifest_file.exists():
            self.log(f"Manifest file not found: {manifest_file}", 'error')
            return False
        
        all_valid = True
        line_num = 0
        
        try:
            with open(manifest_file, 'r') as f:
                for line in f:
                    line_num += 1
                    if line.strip():
                        try:
                            entry = json.loads(line)
                            all_valid &= self.validate_no_nan_inf(entry, f"{manifest_file}:L{line_num}")
                            all_valid &= self.validate_manifest_schema(entry)
                        except json.JSONDecodeError as e:
                            self.log(f"JSON decode error in {manifest_file} line {line_num}: {e}", 'error')
                            all_valid = False
        except Exception as e:
            self.log(f"Failed to read manifest {manifest_file}: {e}", 'error')
            return False
        
        if all_valid:
            self.log(f"Manifest validation passed for {manifest_file} ({line_num} entries)")
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(description='Update_19 Integrity Validator')
    parser.add_argument('paths', nargs='+', help='Result files or manifest files to validate')
    parser.add_argument('--retro-fix', action='store_true', help='Apply retro-fix to result files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    validator = Update19Validator(verbose=args.verbose)
    
    for path_str in args.paths:
        path = Path(path_str)
        
        if path.suffix == '.json':
            # Result file
            print(f"\nValidating result file: {path}")
            
            if args.retro_fix:
                print("Applying retro-fix...")
                try:
                    fixed_data = retro_fix_timing(path)
                    # Write back the fixed data
                    backup_path = path.with_suffix('.json.backup')
                    path.rename(backup_path)
                    with open(path, 'w') as f:
                        json.dump(fixed_data, f, indent=2, default=str)
                    print(f"Retro-fixed {path}, backup saved to {backup_path}")
                except Exception as e:
                    print(f"Retro-fix failed for {path}: {e}")
            
            validator.validate_result_file(path)
            
        elif path.suffix == '.jsonl':
            # Manifest file
            print(f"\nValidating manifest file: {path}")
            validator.validate_manifest_file(path)
        
        else:
            print(f"Unknown file type: {path}")
    
    # Summary
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Errors: {len(validator.errors)}")
    print(f"Warnings: {len(validator.warnings)}")
    
    if validator.errors:
        print("\nERRORS:")
        for error in validator.errors:
            print(f"  - {error}")
    
    if validator.warnings:
        print("\nWARNINGS:")
        for warning in validator.warnings:
            print(f"  - {warning}")
    
    return 0 if not validator.errors else 1


if __name__ == '__main__':
    exit(main())