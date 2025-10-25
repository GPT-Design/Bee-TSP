#!/usr/bin/env python3
"""
Evaluation Tripwires - Update_17.txt Runtime Enforcement
Prevents synthetic/non-TSPLIB evaluation paths when STRICT_TSPLIB_EVAL=1
"""

import os
import sys
from typing import Any


class SyntheticEvaluationError(Exception):
    """Raised when synthetic evaluation is attempted with STRICT_TSPLIB_EVAL=1"""
    pass


def check_strict_eval_env():
    """
    Check if STRICT_TSPLIB_EVAL=1 is set.
    Returns True if strict evaluation is enforced.
    """
    return os.environ.get("STRICT_TSPLIB_EVAL") == "1"


def tripwire_guard(evaluator_type: str, function_name: str = None):
    """
    Runtime tripwire guard against synthetic evaluation paths.
    
    Args:
        evaluator_type: Type of evaluator being used (e.g., "synthetic", "normalized", "float")
        function_name: Optional function name for better error messages
    
    Raises:
        SyntheticEvaluationError: If STRICT_TSPLIB_EVAL=1 and synthetic evaluator detected
    """
    if not check_strict_eval_env():
        return  # Allow synthetic paths when not in strict mode
    
    banned_types = {
        'synthetic', 'mock', 'dummy', 'stub', 'normalized', 'scaled',
        'approx', 'float_len', 'fast_eval', 'toy', 'test_eval', 'sampled_len', 'fake'
    }
    
    if evaluator_type.lower() in banned_types:
        error_msg = f"TRIPWIRE VIOLATION: {evaluator_type} evaluator forbidden with STRICT_TSPLIB_EVAL=1"
        if function_name:
            error_msg += f" in {function_name}()"
        error_msg += ". Only TSPLIB integer evaluation allowed."
        
        raise SyntheticEvaluationError(error_msg)


def enforce_tsplib_only_import(module_name: str):
    """
    Tripwire for import-time checks.
    Call this when importing any evaluation-related module.
    """
    if not check_strict_eval_env():
        return
    
    banned_modules = {
        'synthetic_evaluator', 'mock_evaluator', 'normalized_evaluator',
        'float_evaluator', 'fast_evaluator', 'approx_evaluator'
    }
    
    if module_name.lower() in banned_modules:
        raise SyntheticEvaluationError(
            f"IMPORT VIOLATION: Module '{module_name}' forbidden with STRICT_TSPLIB_EVAL=1. "
            f"Only TSPLIB integer evaluation modules allowed."
        )


def log_evaluation_source(eval_source: str, controller_len_int: int, 
                         reported_len_raw: Any = None) -> dict:
    """
    Log evaluation with paranoia fields per Update_17.txt section 8.
    
    Args:
        eval_source: Source of evaluation (should be "TSPLIB_INTEGER")
        controller_len_int: Integer length from controller TSPLIB re-sum
        reported_len_raw: Raw length reported by bridge/miner (optional)
    
    Returns:
        Dict with paranoia fields for logging
    """
    delta_ok = True
    
    if reported_len_raw is not None:
        try:
            reported_int = int(reported_len_raw)
            delta_ok = (reported_int == controller_len_int)
        except (ValueError, TypeError):
            delta_ok = False
    
    return {
        "eval_source": eval_source,
        "controller_len_int": controller_len_int,
        "reported_len_raw": reported_len_raw,
        "delta_ok": delta_ok,
        "strict_eval_enforced": check_strict_eval_env()
    }


def validate_tour_first_acceptance(tour: Any, controller_len_int: int, 
                                  reported_len: Any = None) -> bool:
    """
    Validate TOUR-FIRST acceptance per Update_17.txt section 2.
    
    Args:
        tour: Candidate tour (must not be None/empty)
        controller_len_int: Integer length from TSPLIB re-sum
        reported_len: Reported length (ignored per TOUR-FIRST policy)
    
    Returns:
        bool: True if tour is valid for acceptance
    
    Raises:
        SyntheticEvaluationError: If tour is invalid
    """
    if tour is None or len(tour) == 0:
        raise SyntheticEvaluationError(
            "TOUR-FIRST VIOLATION: no tour provided. Reported lengths are ignored."
        )
    
    if not isinstance(controller_len_int, int):
        raise SyntheticEvaluationError(
            f"CONTROLLER VIOLATION: controller_len_int must be integer, got {type(controller_len_int)}"
        )
    
    return True


def setup_evaluation_environment():
    """
    Setup and validate evaluation environment per Update_17.txt section 3.
    Call this at program startup.
    """
    print("=== EVALUATION ENVIRONMENT SETUP ===")
    
    # Check STRICT_TSPLIB_EVAL
    strict_eval = os.environ.get("STRICT_TSPLIB_EVAL")
    if strict_eval == "1":
        print("OK STRICT_TSPLIB_EVAL=1 - Only integer TSPLIB evaluation allowed")
    else:
        print(f"WARN STRICT_TSPLIB_EVAL='{strict_eval}' - Synthetic paths may be allowed")
    
    # Check threading environment
    threading_vars = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']
    for var in threading_vars:
        value = os.environ.get(var)
        if value == "1":
            print(f"OK {var}=1 - Single-thread enforced")
        else:
            print(f"WARN {var}='{value}' - Multi-threading may affect determinism")
    
    # Install import hook for synthetic evaluator detection
    if check_strict_eval_env():
        print("OK Strict evaluation tripwires ACTIVE")
        _install_import_hook()
    else:
        print("INFO Strict evaluation tripwires INACTIVE")
    
    print("=== EVALUATION ENVIRONMENT READY ===")


def _install_import_hook():
    """Install import hook to catch synthetic evaluator imports."""
    import importlib.util
    
    original_find_spec = importlib.util.find_spec
    
    def hooked_find_spec(name, package=None):
        if name and any(banned in name.lower() for banned in 
                       ['synthetic', 'mock', 'dummy', 'normalized', 'float_eval']):
            raise SyntheticEvaluationError(
                f"IMPORT HOOK VIOLATION: Module '{name}' forbidden with STRICT_TSPLIB_EVAL=1"
            )
        return original_find_spec(name, package)
    
    importlib.util.find_spec = hooked_find_spec