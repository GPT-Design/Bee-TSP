# bee_tsp package initializer
# Tries the standard solver filename first, then a fallback.
try:
    from .solver import BeeTSPSolver  # preferred
except Exception:
    try:
        from .solver_large_patch import BeeTSPSolver  # fallback
    except Exception as e:
        raise ImportError(
            "bee_tsp: could not import BeeTSPSolver from solver.py "
            "or solver_large_patch.py"
        ) from e

__all__ = ["BeeTSPSolver"]
