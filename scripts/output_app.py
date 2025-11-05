#!/usr/bin/env python3
"""
Standardized Output Manager for Bee-TSP Solver - Update_20 Compliant
Implements the naming convention: YY_M_DD_HHMM_<instance_name>.<extension>
All results are centralized in the \results subdirectory.
Includes global manifest, timezone support, and CI validation hooks.
"""

import os
import json
from random import seed
import threading
import math
from datetime import datetime, timezone as _dt_timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
from tools.tour_io import write_tsplib_tour, sha256_file, is_perm_0n
# Python 3.9+ has zoneinfo; otherwise use backport (Windows often needs tzdata too)
# Prefer stdlib zoneinfo; fall back to backport if available; else None
from datetime import tzinfo as _dt_tzinfo, timezone as _dt_timezone
try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except Exception:
    _ZoneInfo = None  # zoneinfo not available

try:
    import yaml
except Exception:
    yaml = None

# at top of scripts/output_app.py
try:
    from tools.tour_io import write_tsplib_tour, sha256_file, is_perm_0n
except Exception:
    write_tsplib_tour = None  # type: ignore
    sha256_file = None        # type: ignore
    is_perm_0n = None         # type: ignore

# Global lock for manifest writes
_manifest_lock = threading.Lock()

class StandardOutputManager:
    """Manages standardized output paths and file operations for TSP solver results."""
    
    def __init__(self, instance_name: str, base_results_dir: str = "results", 
                 custom_timestamp: Optional[datetime] = None):
        """
        Initialize the output manager.
        
        Args:
            instance_name: TSPLIB instance name (without .tsp extension)
            base_results_dir: Base directory for all results (default: "results")
            custom_timestamp: Override timestamp (for testing/reproducibility)
        """
        self.instance_name = self._clean_instance_name(instance_name)
        self.base_results_dir = Path(base_results_dir)
        self.seed: Optional[int] = None 
        
        # Handle timezone per Update_20 specs
        self.timezone = self._get_output_timezone()
        if custom_timestamp is not None:
            self.timestamp = custom_timestamp
        else:
            self.timestamp = datetime.now(self.timezone) if self.timezone else datetime.now(_dt_timezone.utc)

        self.timestamp_str = self._format_timestamp()
        self.base_filename = f"{self.timestamp_str}_{self.instance_name}"
        
        # Create results directory structure
        self._ensure_directories()
        
        # Global manifest path
        self.global_manifest_path = self.base_results_dir / "manifest.jsonl"
    
    def set_seed(self, seed: int) -> None:
        """Declare the active seed for per-seed artifacts/logs."""
        self.seed = int(seed)

    def _require_seed(self) -> int:
        if self.seed is None:
            raise AttributeError("seed not set; call set_seed(seed) before per-seed outputs")
        return self.seed

    def _clean_instance_name(self, name: str) -> str:
        """Clean instance name to remove .tsp extension and invalid characters."""
        if name.endswith('.tsp'):
            name = name[:-4]
        # Remove any path components
        name = os.path.basename(name)
        # Replace invalid filename characters
        name = name.replace(' ', '_').replace('-', '_')
        return name
    
    def _get_output_timezone(self) -> Optional[_dt_tzinfo]:
        """Update_20: pick output TZ (default Australia/Adelaide, override via OUTPUT_TZ)."""
        tz_name = os.environ.get("OUTPUT_TZ", "Australia/Adelaide")
        if _ZoneInfo is None:
            return None
        try:
            return _ZoneInfo(tz_name)
        except Exception:
            try:
                return _ZoneInfo("UTC")
            except Exception:
                return None
    
    def _format_timestamp(self) -> str:
        """Format as YY_MM_DD_HHMM in the configured output timezone."""
        dt = self.timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt_timezone.utc)

        tz = self._get_output_timezone()
        if tz is not None:
            try:
                dt = dt.astimezone(tz)
            except Exception:
                dt = dt.astimezone(_dt_timezone.utc)

        # yy_mm_dd_hhmm (zero-pad month/day/hour/minute)
        return f"{dt.year % 100:02d}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}"
    
    def _ensure_directories(self):
        """Create necessary directory structure."""
        # Main results directory
        self.base_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run-specific subdirectory
        self.run_dir = self.base_results_dir / self.base_filename
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Archives directory for old runs
        self.archive_dir = self.base_results_dir / "archives"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def _handle_collision(self, path: Path) -> Path:
        """Handle filename collisions by appending suffix."""
        if not path.exists():
            return path
        
        counter = 1
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        
        while counter < 1000:  # Safety limit
            new_name = f"{stem}_{counter:03d}{suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
        
        raise RuntimeError(f"Too many filename collisions for {path}")
    
    def get_jsonl_path(self) -> Path:
        ts = self._format_timestamp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed = self._require_seed()
        return self.run_dir / f"seed_{seed}_{ts}.jsonl"

    def get_tour_path(self) -> Path:
        ts = self._format_timestamp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        seed = self._require_seed()
        return self.run_dir / f"seed_{seed}_{ts}.tour"

    def get_results_path(self) -> Path:
        """Run summary (.json)."""
        ts = self._format_timestamp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / f"{self.base_filename}_{ts}.json"

    def get_config_path(self) -> Path:
        """Config snapshot (.yaml)."""
        ts = self._format_timestamp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / f"{self.base_filename}_{ts}.yaml"

    def get_log_path(self, seed: int) -> Path:
        ts = self._format_timestamp()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir / f"seed_{seed}_{ts}.jsonl"
    
    def get_run_directory(self) -> Path:
        """Get the run-specific directory path."""
        return self.run_dir
    
    def save_results(self, results_data: Dict[str, Any]) -> Path:
        """
        Save results data to standardized JSON file with Update_20 schema compliance.
        
        Args:
            results_data: Dictionary containing results metrics
            
        Returns:
            Path to saved results file
        """
        results_path = self.get_results_path()
        
        # Ensure Update_20 compliant schema - minimal required fields
        enhanced_data = {
            "instance": self.instance_name,
            "timestamp": self.timestamp.isoformat(),
            "output_format_version": "2.0",  # Update_20 version
            **results_data
        }
        
        # Validate no NaNs/Infinities per Update_20
        self._validate_no_nans(enhanced_data, "results_data")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        return results_path

    # inside class StandardOutputManager
    def _calculate_tour_sha256(self, tour_path: Path) -> str:
        """Return SHA256 of the given file (streamed, 1 MiB chunks)."""
        import hashlib
        h = hashlib.sha256()
        with open(tour_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    
    def save_config(self, config_data: Dict[str, Any]) -> Path:
        """
        Save configuration data. Prefer YAML; fall back to JSON if PyYAML unavailable.
        """
        # Add run metadata
        enhanced_config = {
            **config_data,
            "run_metadata": {
                "instance_name": self.instance_name,
                "timestamp": self.timestamp.isoformat(),
                "output_format_version": "1.0",
                "run_directory": str(self.run_dir),
            },
        }

        if yaml is None:
            # Fallback: write JSON next to where YAML would be
            json_path = self.get_config_path().with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
            return json_path

        # Preferred: YAML
        config_path = self.get_config_path()
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(enhanced_config, f, sort_keys=False, default_flow_style=False)
        return config_path

    def _validate_no_nans(self, data: Any, path: str = "") -> None:
        """Validate that data contains no NaNs or Infinities."""
        if isinstance(data, dict):
            for key, value in data.items():
                self._validate_no_nans(value, f"{path}.{key}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                self._validate_no_nans(item, f"{path}[{i}]")
        elif isinstance(data, float):
            if math.isnan(data) or math.isinf(data):
                raise ValueError(f"NaN/Infinity found at {path}: {data}")
    
    # replace your current save_tour with this
    def save_tour(self, tour: list, dimension: int, tour_length: Optional[int] = None,
                write_sha_sidecar: bool = False) -> Path:
        """
        Save tour in TSPLIB format to standardized .tour file.
        If write_sha_sidecar=True, also writes seed_... .tour.sha256 next to it.
        """
        tour_path = self.get_tour_path()

        with open(tour_path, 'w', encoding='utf-8') as f:
            f.write(f"NAME : {self.instance_name}\n")
            f.write("TYPE : TOUR\n")
            f.write(f"DIMENSION : {dimension}\n")
            if tour_length is not None:
                f.write(f"COMMENT : Bee-TSP Mark1; length={int(tour_length)}\n")
            else:
                f.write("COMMENT : Bee-TSP Mark1\n")
            f.write("TOUR_SECTION\n")
            # Write 1-based IDs (accepts either 0-based or 1-based input)
            zero_based = (min(tour) == 0)
            for city in tour:
                f.write(f"{(city + 1) if zero_based else city}\n")
            f.write("-1\nEOF\n")

        # Always compute hash for the manifest, but only write sidecar if asked
        _sha = self._calculate_tour_sha256(tour_path)
        if write_sha_sidecar:
            sha_path = tour_path.with_suffix('.tour.sha256')
            with open(sha_path, 'w', encoding='utf-8') as sf:
                sf.write(f"{_sha}  {tour_path.name}\n")

        return tour_path
    
    def save_seed_log(self, seed: int, events: list) -> Path:
        """
        Save individual seed events to JSONL format.
        
        Args:
            seed: Random seed used
            events: List of event dictionaries
            
        Returns:
            Path to saved log file
        """
        log_path = self.get_log_path(seed)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            for event in events:
                # Ensure each event has seed metadata
                event_with_seed = {"seed": seed, **event}
                f.write(json.dumps(event_with_seed, ensure_ascii=False) + '\n')
        
        return log_path
    
    def append_to_global_manifest(self, manifest_entry: Dict[str, Any]) -> None:
        """
        Append entry to global manifest.jsonl with thread safety.
        
        Args:
            manifest_entry: Dictionary with exact Update_20 manifest schema
        """
        # Validate manifest entry schema per Update_20
        required_fields = {
            "instance", "metric", "mode", "seed", "wall_s", "elapsed_s", 
            "epochs_completed", "epoch_s", "tour_path", "tour_sha256", 
            "length_tsplib", "opt", "best_known", "parity", "ttt10_s", 
            "ttt5_s", "early_slope", "improvements_per_min", "hive_attrib", 
            "timestamp_utc"
        }
        
        missing_fields = required_fields - set(manifest_entry.keys())
        if missing_fields:
            raise ValueError(f"Missing required manifest fields: {missing_fields}")
        
        # Validate no NaNs/Infinities
        self._validate_no_nans(manifest_entry, "manifest_entry")
        
        # Thread-safe append to global manifest
        with _manifest_lock:
            with open(self.global_manifest_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
    
    def create_manifest_entry(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a manifest entry (Update_20 schema) with deterministic, this-run-only
        tour emission. It first looks for the canonical file for THIS seed/run and
        uses it if present; otherwise it writes from the provided best_tour.
        """
        # --- Robust UTC timestamp ---
        try:
            ts_utc = self.timestamp.astimezone(_dt_timezone.utc).isoformat()
        except Exception:
            ts_utc = self.timestamp.replace(tzinfo=_dt_timezone.utc).isoformat()

        # Ensure per-seed path (so get_tour_path() is correct)
        self.set_seed(int(run_data.get("seed", 0)))
        tpath = self.get_tour_path()  # Path

        # Pull fields we’ll pass through
        inst_name = run_data.get("instance", self.instance_name) or "<unknown>"
        metric    = run_data.get("metric", "EUC_2D")

        def _safe_int(x, default=None):
            try:
                return int(x)
            except Exception:
                return default

        best_tour_0b = run_data.get("best_tour")
        length_int   = _safe_int(run_data.get("best_length"))

        tour_path_value: Optional[str] = None
        tour_sha_value:  Optional[str] = None

        try:
            # 1) Prefer an existing file at the canonical path for THIS run
            if tpath.exists() and tpath.is_file():
                tour_path_value = str(tpath)
                tour_sha_value  = self._calculate_tour_sha256(tpath)
            # 2) Otherwise, try to write it from in-memory data
            elif best_tour_0b and length_int is not None:
                try:
                    bt_list = list(best_tour_0b)  # handles numpy arrays too
                except Exception:
                    bt_list = None
                if bt_list and len(bt_list) > 0:
                    # Primary attempt: with kwarg
                    try:
                        written = self.save_tour(bt_list, len(bt_list),
                                                tour_length=length_int,
                                                write_sha_sidecar=False)
                    except TypeError:
                        # Fallback if signature doesn’t accept that kwarg
                        written = self.save_tour(bt_list, len(bt_list),
                                                tour_length=length_int)
                    tour_path_value = str(written)
                    tour_sha_value  = self._calculate_tour_sha256(written)
        except Exception as e:
            print(f"[MANIFEST] emission error: {e!r}")
            tour_path_value = None
            tour_sha_value  = None

        manifest_entry = {
            "instance": inst_name,
            "metric": metric,
            "mode": str(run_data.get("mode", "SH")),
            "seed": int(run_data.get("seed", 0)),
            "wall_s": float(run_data.get("wall_s", 0)),
            "elapsed_s": float(run_data.get("elapsed_s", 0.0)),
            "epochs_completed": int(run_data.get("epochs_completed", 0)),
            "epoch_s": int(run_data.get("epoch_s", 0)),
            "tour_path": tour_path_value,
            "tour_sha256": tour_sha_value,
            "length_tsplib": _safe_int(run_data.get("length_tsplib"), 0),
            "opt": run_data.get("opt", None),
            "best_known": run_data.get("best_known", None),
            "parity": float(run_data.get("parity", 0.0)),
            "ttt10_s": run_data.get("ttt10_s", None),
            "ttt5_s": run_data.get("ttt5_s", None),
            "early_slope": run_data.get("early_slope", None),
            "improvements_per_min": run_data.get("improvements_per_min", None),
            "hive_attrib": run_data.get("hive_attrib", None),
            "timestamp_utc": ts_utc,
            "init_mix": run_data.get("init_mix", None),
            "candidate_caps": run_data.get("candidate_caps", None),
            "hk": run_data.get("hk", None),
        }
        return manifest_entry
    
    def get_summary_info(self) -> Dict[str, Any]:
        """Get summary information about this run's output setup."""
        return {
            "instance_name": self.instance_name,
            "timestamp": self.timestamp.isoformat(),
            "timestamp_format": self.timestamp_str,
            "timezone": str(self.timezone),
            "base_filename": self.base_filename,
            "run_directory": str(self.run_dir),
            "tour_path": str(self.get_tour_path()),
            "results_path": str(self.get_results_path()),
            "config_path": str(self.get_config_path()),
            "global_manifest": str(self.global_manifest_path)
        }
    
    @staticmethod
    def parse_timestamp_from_filename(filename: str) -> Tuple[Optional[datetime], str]:
        """
        Parse timestamp and instance name from standardized filename.
        
        Args:
            filename: Filename in format YY_M_DD_HHMM_instance.ext
            
        Returns:
            Tuple of (datetime object, instance_name) or (None, original_string)
        """
        try:
            stem = Path(filename).stem
            parts = stem.split('_')
            if len(parts) >= 5:
                yy, m, dd, hhmm = parts[:4]
                instance = '_'.join(parts[4:])
                
                year = 2000 + int(yy)
                month = int(m)
                day = int(dd)
                hour = int(hhmm[:2])
                minute = int(hhmm[2:])
                
                dt = datetime(year, month, day, hour, minute)
                return dt, instance
        except (ValueError, IndexError):
            pass
        
        return None, filename
    
    @classmethod
    def from_existing_file(cls, filepath: str) -> 'StandardOutputManager':
        """
        Create StandardOutputManager from existing standardized filename.
        
        Args:
            filepath: Path to existing file with standardized name
            
        Returns:
            StandardOutputManager instance configured for that run
        """
        filename = os.path.basename(filepath)
        timestamp, instance_name = cls.parse_timestamp_from_filename(filename)
        
        if timestamp is None:
            raise ValueError(f"Cannot parse standardized filename: {filename}")
        
        results_dir = Path(filepath).parent.parent if Path(filepath).parent.name.startswith(('25_', '24_', '26_')) else "results"
        
        return cls(instance_name, str(results_dir), timestamp)


def migrate_legacy_output(legacy_path: str, instance_name: str) -> StandardOutputManager:
    """
    Migrate legacy output file to standardized format.
    
    Args:
        legacy_path: Path to existing non-standard file
        instance_name: TSPLIB instance name
        
    Returns:
        StandardOutputManager for the migrated file
    """
    mgr = StandardOutputManager(instance_name)
    legacy_file = Path(legacy_path)
    
    if legacy_file.suffix == '.json':
        # Copy JSON results file
        if legacy_file.exists():
            with open(legacy_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            mgr.save_results(data)
    
    elif legacy_file.suffix == '.yaml':
        # Copy YAML config file  
        if legacy_file.exists():
            with open(legacy_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            mgr.save_config(data)
    
    return mgr


# Update_20 CI Validation Functions
def validate_no_nans_in_file(filepath: Path) -> bool:
    """Validate that a JSON/JSONL file contains no NaN/Infinity values."""
    try:
        # Create temporary instance to access the validation method
        temp_mgr = StandardOutputManager("temp")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            temp_mgr._validate_no_nans(data, str(filepath))
        elif filepath.suffix == '.jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data = json.loads(line)
                        temp_mgr._validate_no_nans(data, f"{filepath}:{line_num}")
        return True
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Validation failed for {filepath}: {e}")
        return False

def validate_time_plausibility(results_data: Dict[str, Any]) -> bool:
    """Validate time integrity per Update_19/20 specs."""
    try:
        elapsed_s = float(results_data.get('elapsed_s', 0))
        wall_s = float(results_data.get('wall_s', 0))
        epochs_completed = int(results_data.get('epochs_completed', 0))
        epoch_s = int(results_data.get('epoch_s', 0))
        
        elapsed_from_epochs_s = epochs_completed * epoch_s
        
        # Integrity check per Update_19
        if wall_s > 0:
            deviation = abs(elapsed_s - min(wall_s, elapsed_from_epochs_s)) / wall_s
            if deviation > 0.02:  # 2% tolerance
                print(f"Time integrity FAIL: deviation {deviation:.3f} > 0.02")
                return False
        
        return True
    except (ValueError, KeyError) as e:
        print(f"Time validation failed: {e}")
        return False

def validate_tour_artifact(tour_path: Path, expected_sha256: str, expected_length: int) -> bool:
    """Validate tour file SHA256 and length per Update_20."""
    if not tour_path.exists():
        print(f"Tour file missing: {tour_path}")
        return False
    
    try:
        # Verify SHA256
        mgr = StandardOutputManager("temp")
        actual_sha256 = mgr._calculate_tour_sha256(tour_path)
        if actual_sha256 != expected_sha256:
            print(f"SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}")
            return False
        
        # TODO: Verify TSPLIB length re-calculation
        # This would require loading the instance and re-calculating the tour length
        print(f"Tour artifact validation passed for {tour_path}")
        return True
        
    except Exception as e:
        print(f"Tour validation failed: {e}")
        return False

def validate_manifest_schema(manifest_entry: Dict[str, Any]) -> bool:
    """Validate manifest entry has required Update_20 schema fields."""
    required_fields = {
        "instance", "metric", "mode", "seed", "wall_s", "elapsed_s", 
        "epochs_completed", "epoch_s", "tour_path", "tour_sha256", 
        "length_tsplib", "opt", "best_known", "parity", "ttt10_s", 
        "ttt5_s", "early_slope", "improvements_per_min", "hive_attrib", 
        "timestamp_utc"
    }
    
    missing_fields = required_fields - set(manifest_entry.keys())
    if missing_fields:
        print(f"Missing required manifest fields: {missing_fields}")
        return False
    
    # Validate types
    try:
        assert isinstance(manifest_entry["instance"], str)
        assert isinstance(manifest_entry["metric"], str)
        assert isinstance(manifest_entry["mode"], str)
        assert isinstance(manifest_entry["seed"], int)
        assert isinstance(manifest_entry["wall_s"], int)
        assert isinstance(manifest_entry["elapsed_s"], (int, float))
        assert isinstance(manifest_entry["epochs_completed"], int)
        assert isinstance(manifest_entry["epoch_s"], int)
        assert isinstance(manifest_entry["length_tsplib"], int)
        assert isinstance(manifest_entry["parity"], str)
        # null values are OK for optional fields
        return True
    except AssertionError as e:
        print(f"Manifest schema type validation failed: {e}")
        return False

def check_banned_synthetic_strings(filepath: Path, banned_strings: List[str] = None) -> bool:
    """Check file for banned synthetic/mock strings per Update_17."""
    if banned_strings is None:
        banned_strings = ["mock", "synthetic", "fake", "dummy", "test_data"]
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().lower()
        
        for banned in banned_strings:
            if banned.lower() in content:
                print(f"Banned string '{banned}' found in {filepath}")
                return False
        return True
    except Exception as e:
        print(f"Failed to check banned strings in {filepath}: {e}")
        return False

def run_full_ci_validation(run_directory: Path) -> bool:
    """Run complete CI validation suite on a run directory."""
    print(f"Running CI validation on {run_directory}")
    
    all_passed = True
    
    # Find JSON files and validate no NaNs
    for json_file in run_directory.glob("*.json"):
        if not validate_no_nans_in_file(json_file):
            all_passed = False
    
    # Find JSONL files and validate no NaNs  
    for jsonl_file in run_directory.glob("*.jsonl"):
        if not validate_no_nans_in_file(jsonl_file):
            all_passed = False
    
    # Find main results file and validate time plausibility
    results_files = list(run_directory.glob("*.json"))
    if results_files:
        main_results = results_files[0]  # Should be only one
        try:
            with open(main_results, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            if not validate_time_plausibility(results_data):
                all_passed = False
        except Exception as e:
            print(f"Failed to validate time plausibility: {e}")
            all_passed = False
    
    # Check for banned strings in all text files
    for text_file in run_directory.glob("*"):
        if text_file.suffix in ['.json', '.jsonl', '.yaml', '.tour']:
            if not check_banned_synthetic_strings(text_file):
                all_passed = False
    
    print(f"CI validation {'PASSED' if all_passed else 'FAILED'} for {run_directory}")
    return all_passed


if __name__ == "__main__":
    # Example usage and testing
    print("Testing StandardOutputManager...")
    
    # Test basic functionality
    mgr = StandardOutputManager("pla33810")
    print(f"Summary: {json.dumps(mgr.get_summary_info(), indent=2)}")
    
    # Test file operations
    test_results = {
        "best_length": 12345,
        "wall_time_s": 3600,
        "gap_to_optimal_pct": 2.5,
        "seeds_completed": [42, 123, 456],
        "parity": "OK"
    }
    
    test_config = {
        "solver": {
            "candidate": {"k": 40},
            "termination": {"wall_time_s": 3600}
        }
    }
    
    results_path = mgr.save_results(test_results)
    config_path = mgr.save_config(test_config)
    tour_path = mgr.save_tour([1, 3, 2, 5, 4], dimension=5, tour_length=100)
    
    print(f"Files created:")
    print(f"  Results: {results_path}")
    print(f"  Config: {config_path}")
    print(f"  Tour: {tour_path}")
    
    # Test filename parsing
    dt, inst = StandardOutputManager.parse_timestamp_from_filename("25_9_12_1805_pla33810.json")
    print(f"Parsed: {dt}, {inst}")