
from __future__ import annotations
from typing import List, Tuple, Callable, Optional
import math
from pathlib import Path

def _round_tsp(x: float) -> int:
    return int(round(x))

def _euc2d_dist(coords: List[Tuple[float,float]]):
    def d(i: int, j: int) -> int:
        if i == j: return 0
        xi, yi = coords[i]; xj, yj = coords[j]
        return _round_tsp(((xi - xj)**2 + (yi - yj)**2) ** 0.5)
    return d

def _matrix_dist(mat):
    def d(i: int, j: int) -> int:
        return mat[i][j]
    return d

def load_tsplib(path: str):
    name = None; n = None
    ewt = None; ewf = None
    coords = []
    matrix_vals = []
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line: continue
        up = line.upper()
        if up.startswith("NAME"):
            name = line.split(":",1)[1].strip() if ":" in line else line.split()[1]
        elif up.startswith("DIMENSION"):
            n = int(line.split(":",1)[1].strip())
        elif up.startswith("EDGE_WEIGHT_TYPE"):
            ewt = line.split(":",1)[1].strip().upper()
        elif up.startswith("EDGE_WEIGHT_FORMAT"):
            ewf = line.split(":",1)[1].strip().upper()
        elif up.startswith("NODE_COORD_SECTION"):
            for _ in range(n or 0):
                parts = lines[i].strip().split(); i += 1
                if len(parts) < 3: break
                coords.append((float(parts[-2]), float(parts[-1])))
        elif up.startswith("EDGE_WEIGHT_SECTION"):
            # Expect FULL_MATRIX only (minimal support)
            count = (n or 0) * (n or 0)
            while i < len(lines):
                s = lines[i].strip(); i += 1
                if s.upper().startswith("EOF") or s.upper().startswith("NODE_COORD_SECTION") or s.upper().startswith("DISPLAY"):
                    i -= 1; break
                if not s: continue
                for tok in s.split():
                    matrix_vals.append(int(float(tok)))
        elif up.startswith("EOF"):
            break
    if n is None:
        raise ValueError("DIMENSION missing")
    if coords:
        dist = _euc2d_dist(coords)
    elif matrix_vals:
        if len(matrix_vals) != n*n:
            raise ValueError("Only EXPLICIT FULL_MATRIX is supported in minimal loader")
        mat = [matrix_vals[r*n:(r+1)*n] for r in range(n)]
        dist = _matrix_dist(mat)
    else:
        raise ValueError("Unsupported TSPLIB format for minimal loader")
    return {"name": name or Path(path).stem, "n": n, "coords": coords if coords else None, "dist": dist}
