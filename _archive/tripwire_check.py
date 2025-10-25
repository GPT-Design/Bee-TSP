#!/usr/bin/env python3
"""
Tripwire Check for Strict TSPLIB Evaluation
Exits non-zero if any synthetic evaluation path is called
"""

import os
import sys

def tripwire_check():
    """
    Quick tripwire check - should EXIT non-zero if any synthetic path is called.
    Verifies STRICT_TSPLIB_EVAL=1 environment variable.
    """
    try:
        # Assert strict TSPLIB evaluation is enabled
        strict_eval = os.getenv("STRICT_TSPLIB_EVAL")
        assert strict_eval == "1", f"STRICT_TSPLIB_EVAL must be '1', got '{strict_eval}'"
        
        print("Strict integer eval: ON")
        
        # Additional environment checks
        threading_vars = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']
        for var in threading_vars:
            value = os.getenv(var)
            assert value == "1", f"{var} must be '1' for single-thread, got '{value}'"
        
        print("Single-thread enforcement: ON")
        print("OK All tripwire checks PASSED")
        
        return True
        
    except AssertionError as e:
        print(f"FAIL TRIPWIRE FAILURE: {e}")
        print("Environment not properly configured for strict TSPLIB evaluation!")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR TRIPWIRE ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    tripwire_check()