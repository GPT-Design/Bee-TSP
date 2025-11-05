# bee_tsp/features.py (Mark-1)
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING
import math, time, random

# Optional deps guarded
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from scipy.spatial import cKDTree as _KDTree  # type: ignore
    _HAS_KDTREE = True
except Exception:
    _KDTree = None  # type: ignore
    _HAS_KDTREE = False

# --- Type-only deps (never executed at runtime) ---
if TYPE_CHECKING:
    import numpy as tnp
    from numpy.typing import NDArray as TNDArray
else:
    tnp = None            # type: ignore
    TNDArray = Any        # type: ignore

# Canonical array type for annotations: NDArray at type-check time, Any at runtime
Array = TNDArray

# --------- small helpers ---------
def _to_array(coords) -> Tuple[Optional[Array], int]:
    """Convert coords to a (n,2) float64 ndarray if numpy is available."""
    if not _HAS_NUMPY:
        # best-effort size only
        try:
            return None, len(coords)
        except Exception:
            return None, 0
    try:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or not np.isfinite(arr).all():
            return None, 0
        return arr, int(arr.shape[0])
    except Exception:
        return None, 0

def _bbox(arr: Array) -> Tuple[float, float, float, float]:
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1])

def _principal_anisotropy(arr: "Array.ndarray") -> Optional[float]:
    """sqrt(lambda_max / lambda_min) from covariance eigenvalues. None if degenerate."""
    try:
        C = np.cov(arr.T)  # 2x2
        w, _ = np.linalg.eig(C)
        w = np.sort(np.real(w))
        lam_min = max(float(w[0]), 0.0)
        lam_max = max(float(w[1]), 0.0)
        if lam_min <= 1e-15:
            return None
        return math.sqrt(lam_max / lam_min)
    except Exception:
        return None

def _nn_stats(arr: Array, time_budget_ms: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Return (mean_nn, median_nn, cv_nn). Uses KDTree when available, falls back to chunked NumPy.
    cv_nn = std/mean of 1-NN distances (coarse clustering hint).
    """
    t0 = time.time()
    n = arr.shape[0]
    if n <= 1:
        return None, None, None

    # Prefer KDTree
    if _HAS_KDTREE:
        try:
            tree = _KDTree(arr, leafsize=64)
            # query k=2 to skip self; take d[:,1]
            dists, _ = tree.query(arr, k=2, workers=-1)
            nn = np.asarray(dists[:, 1], dtype=np.float64)
            mean = float(np.mean(nn))
            med = float(np.median(nn))
            std = float(np.std(nn))
            cv = None if mean <= 1e-15 else float(std / mean)
            return mean, med, cv
        except Exception:
            pass  # fall through

    # Fallback: approximate via sampling if large
    try:
        if n <= 4000:
            # full O(n^2) in blocks (coarse; still bounded)
            blk = 1024
            mins = np.full(n, np.inf, dtype=np.float64)
            for i0 in range(0, n, blk):
                i1 = min(n, i0 + blk)
                A = arr[i0:i1]
                # compute dists to all; update mins excluding self
                # (A - arr)^2 sum over axis=2
                # memory-aware: do in sub-blocks on columns
                cblk = 4096
                for j0 in range(0, n, cblk):
                    j1 = min(n, j0 + cblk)
                    B = arr[j0:j1]
                    d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
                    if j0 <= i0 < j1:
                        # mask diagonal when overlapping block
                        diag_idx = np.arange(i0, i1) - j0
                        d2[np.arange(i1 - i0), diag_idx] = np.inf
                    mins[i0:i1] = np.minimum(mins[i0:i1], np.sqrt(d2).min(axis=1))
                if (time.time() - t0) * 1000.0 > time_budget_ms:
                    break
            finite = mins[np.isfinite(mins)]
            if finite.size == 0:
                return None, None, None
            mean = float(np.mean(finite))
            med = float(np.median(finite))
            std = float(np.std(finite))
            cv = None if mean <= 1e-15 else float(std / mean)
            return mean, med, cv
        else:
            # sample ~min(4096, n) points
            rng = np.random.default_rng(1234567)
            m = int(min(4096, n))
            idx = rng.choice(n, size=m, replace=False)
            A = arr[idx]
            # query brute force to full arr in chunks
            mins = np.full(m, np.inf, dtype=np.float64)
            cblk = 4096
            for j0 in range(0, n, cblk):
                j1 = min(n, j0 + cblk)
                B = arr[j0:j1]
                d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
                # mask self rows if any overlap (rare in sample)
                mins = np.minimum(mins, np.sqrt(d2).min(axis=1))
                if (time.time() - t0) * 1000.0 > time_budget_ms:
                    break
            finite = mins[np.isfinite(mins)]
            if finite.size == 0:
                return None, None, None
            mean = float(np.mean(finite))
            med = float(np.median(finite))
            std = float(np.std(finite))
            cv = None if mean <= 1e-15 else float(std / mean)
            return mean, med, cv
    except Exception:
        return None, None, None

def _avg_degree_knn(arr: Array, k: int, time_budget_ms: int) -> Optional[float]:
    """
    Symmetric k-NN graph (undirected): degree = number of neighbors in either direction.
    Returns average degree. Fast with KDTree; sampled otherwise.
    """
    t0 = time.time()
    n = arr.shape[0]
    if n == 0 or k <= 0:
        return None
    k = min(k, max(1, n - 1))

    if _HAS_KDTREE:
        try:
            tree = _KDTree(arr, leafsize=64)
            # query k+1 (includes self)
            _, idx = tree.query(arr, k=min(k + 1, n), workers=-1)
            if idx.ndim == 1:  # n==1 corner
                return 0.0
            nbr = idx[:, 1:]  # drop self
            # build symmetric adjacency via a small bitset/integer trick
            deg = np.zeros(n, dtype=np.int32)
            for u in range(n):
                for v in nbr[u]:
                    deg[u] += 1
                    deg[v] += 1
            return float(np.mean(deg))
        except Exception:
            pass

    # fallback: sample approx
    try:
        m = int(min(2048, n))
        rng = random.Random(1234567)
        S = rng.sample(range(n), m)
        deg_sum = 0
        for u in S:
            # brute force k nearest in chunks
            best = []
            cblk = 4096
            for j0 in range(0, n, cblk):
                j1 = min(n, j0 + cblk)
                B = arr[j0:j1]
                d2 = np.sum((arr[u:u+1, :] - B)**2, axis=1)
                for j, d in enumerate(d2, start=j0):
                    if j == u:
                        continue
                    best.append((float(d), int(j)))
            best.sort(key=lambda x: x[0])
            neigh = [v for _, v in best[:k]]
            deg_sum += len(neigh)  # half of symmetric count; still a coarse proxy
            if (time.time() - t0) * 1000.0 > time_budget_ms:
                break
        return float(2.0 * deg_sum / max(1, len(S)))  # symmetrize
    except Exception:
        return None

# --------- public API ----------
def compute_features(
    coords,
    kd_backend: str = "auto",
    _max_threads: int = 1, # reserved for Mark-2; not used in Mark-1
    metric: str = "EUC_2D",
    time_budget_ms: int = 400,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Lightweight, time-boxed instance analyzer.
    Returns a dict of descriptive stats; never raises (ok=False on failure).
    Mark-1 is read-only: the solver does not change behavior based on these.
    """
    t0 = time.time()
    out: Dict[str, Any] = {
        "ok": False, "n": 0, "metric": str(metric),
        "bbox": None, "scale_diag": None,
        "density_est": None,
        "anisotropy": None,
        "mean_nn": None, "median_nn": None, "cv_nn": None,
        "avg_degree_k": None, "k_used": None,
        "kd_backend": None,
        "rng_seed_used": None,
        "elapsed_ms": 0.0,
    }

    # RNG / seed (for any sampling)
    try:
        if seed is None:
            seed = 1234567
        random.seed(seed)
        if _HAS_NUMPY:
            np.random.seed(seed)
        out["rng_seed_used"] = int(seed)
    except Exception:
        pass

    # NumPy guard
    arr, n = _to_array(coords)
    out["n"] = int(n)
    if not _HAS_NUMPY or arr is None or n < 2:
        out["elapsed_ms"] = (time.time() - t0) * 1000.0
        return out  # ok=False

    # KDTree selection
    use_kdtree = False
    if kd_backend == "auto":
        use_kdtree = bool(_HAS_KDTREE)
    elif kd_backend == "scipy":
        use_kdtree = bool(_HAS_KDTREE)
    elif kd_backend == "none":
        use_kdtree = False
    else:
        use_kdtree = bool(_HAS_KDTREE)
    out["kd_backend"] = "kdtree" if use_kdtree else "none"

    # Bounding box, scale, density
    try:
        minx, maxx, miny, maxy = _bbox(arr)
        w = maxx - minx
        h = maxy - miny
        diag = float(math.hypot(w, h))
        area = float(w * h)
        dens = None if area <= 1e-15 else float(n / area)
        out["bbox"] = [minx, maxx, miny, maxy]
        out["scale_diag"] = diag
        out["density_est"] = dens
    except Exception:
        pass

    # Anisotropy via PCA eigenvalue ratio
    if (time.time() - t0) * 1000.0 <= time_budget_ms:
        out["anisotropy"] = _principal_anisotropy(arr)

    # NN stats
    if (time.time() - t0) * 1000.0 <= time_budget_ms:
        mean_nn, med_nn, cv_nn = _nn_stats(arr, time_budget_ms - int((time.time() - t0) * 1000.0))
        out["mean_nn"] = mean_nn
        out["median_nn"] = med_nn
        out["cv_nn"] = cv_nn

    # Avg degree in symmetric k-NN graph (k ~ log2(n)+4, clamped)
    if (time.time() - t0) * 1000.0 <= time_budget_ms:
        k = max(4, min(64, int(math.log2(n)) + 4))
        out["k_used"] = int(k)
        out["avg_degree_k"] = _avg_degree_knn(arr, k, time_budget_ms - int((time.time() - t0) * 1000.0))

    out["elapsed_ms"] = (time.time() - t0) * 1000.0
    out["ok"] = True
    return out
