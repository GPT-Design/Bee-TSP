# bee_tsp/tsplib.py
from __future__ import annotations
import math
from typing import Callable, Dict, List, Tuple

Coord = Tuple[float, float]

def nint(x: float) -> int:
    return int(x + 0.5)

def _geo_to_radians(val: float) -> float:
    # TSPLIB GEO: value is DDD.MM where .MM are minutes (not decimal degrees)
    deg = int(val)
    minutes = val - deg
    return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0  # TSPLIB rule

def make_dist_fn(edge_weight_type: str, coords: List[Coord]) -> Callable[[int, int], int]:
    ewt = edge_weight_type.upper()

    if ewt in ("EUC_2D", "EUC_3D"):
        def d(i: int, j: int) -> int:
            xi, yi = coords[i]; xj, yj = coords[j]
            return nint(math.hypot(xi - xj, yi - yj))
        return d

    if ewt == "CEIL_2D":
        def d(i: int, j: int) -> int:
            xi, yi = coords[i]; xj, yj = coords[j]
            return math.ceil(math.hypot(xi - xj, yi - yj))
        return d

    if ewt in ("MAN_2D", "MAN_3D"):
        def d(i: int, j: int) -> int:
            xi, yi = coords[i]; xj, yj = coords[j]
            return nint(abs(xi - xj) + abs(yi - yj))
        return d

    if ewt in ("MAX_2D", "MAX_3D"):
        def d(i: int, j: int) -> int:
            xi, yi = coords[i]; xj, yj = coords[j]
            return nint(max(abs(xi - xj), abs(yi - yj)))
        return d

    if ewt == "ATT":  # pseudo-Euclidean
        def d(i: int, j: int) -> int:
            xi, yi = coords[i]; xj, yj = coords[j]
            dij = math.sqrt(((xi - xj)**2 + (yi - yj)**2) / 10.0)
            t = nint(dij)
            if t < dij:  # TSPLIB quirk
                t += 1
            return t
        return d

    if ewt == "GEO":
        # Precompute radians
        rad = [(_geo_to_radians(x), _geo_to_radians(y)) for (x, y) in coords]
        R = 6378.388
        def d(i: int, j: int) -> int:
            lat_i, lon_i = rad[i]; lat_j, lon_j = rad[j]
            q1 = math.cos(lon_i - lon_j)
            q2 = math.cos(lat_i - lat_j)
            q3 = math.cos(lat_i + lat_j)
            dij = R * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
            return int(dij)  # truncate
        return d

    raise NotImplementedError(f"EDGE_WEIGHT_TYPE '{edge_weight_type}' not supported")

def load_tsplib(path: str) -> Dict:
    """
    Minimal loader:
      - reads header (NAME, DIMENSION, EDGE_WEIGHT_TYPE, NODE_COORD_SECTION, EDGE_WEIGHT_SECTION, DISPLAY_* ignored)
      - returns {'n', 'coords' or 'weights', 'edge_weight_type', 'dist'}
    """
    header: Dict[str, str] = {}
    coords: List[Coord] = []
    weights: List[List[int]] = []
    ewt = None
    n = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f]

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("NAME"): header["NAME"] = ln.split(":",1)[1].strip()
        elif ln.startswith("TYPE"): header["TYPE"] = ln.split(":",1)[1].strip()
        elif ln.startswith("DIMENSION"):
            n = int(ln.split(":",1)[1])
        elif ln.startswith("EDGE_WEIGHT_TYPE"):
            ewt = ln.split(":",1)[1].strip()
        elif ln.startswith("NODE_COORD_SECTION"):
            i += 1
            for _ in range(n):
                id_, x, y = lines[i].split()[:3]
                coords.append((float(x), float(y)))
                i += 1
            continue
        elif ln.startswith("EDGE_WEIGHT_SECTION"):
            # If explicit matrix is present, read it; distance fn will index this.
            i += 1
            row = []
            while i < len(lines) and lines[i] != "EOF":
                for tok in lines[i].split():
                    row.append(int(float(tok)))
                    if len(row) == n:
                        weights.append(row); row = []
                i += 1
            break
        elif ln.startswith("EOF"):
            break
        i += 1

    if n is None:
        raise ValueError("DIMENSION not found")
    if ewt is None:
        raise ValueError("EDGE_WEIGHT_TYPE not found")

    if coords:
        dist = make_dist_fn(ewt, coords)
        return {"n": n, "coords": coords, "edge_weight_type": ewt, "dist": dist}
    if weights:
        def dist(i: int, j: int) -> int:
            return weights[i][j]
        return {"n": n, "weights": weights, "edge_weight_type": ewt, "dist": dist}
    raise ValueError("No coordinates or explicit edge weights found")
