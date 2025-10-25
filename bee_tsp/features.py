# bee_tsp/features.py
import numpy as np

KD_BACKEND = "numpy"
KDTreeImpl = None
ConvexHull = None

# prefer sklearn if available
try:
    from sklearn.neighbors import KDTree as _SKKD
    KDTreeImpl = _SKKD
    KD_BACKEND = "sklearn"
except Exception:
    try:
        from scipy.spatial import cKDTree as _SPKD
        KDTreeImpl = _SPKD
        KD_BACKEND = "scipy"
    except Exception:
        pass

try:
    from scipy.spatial import ConvexHull as _Hull
    ConvexHull = _Hull
except Exception:
    ConvexHull = None

try:
    from threadpoolctl import threadpool_limits as _tpl
except Exception:
    _tpl = None

def _mean_r1(coords, k=2, max_threads=1):
    n = coords.shape[0]
    if KDTreeImpl is None or n < 3:
        # brute force fallback (O(n^2) but OK for small n)
        diff = coords[:,None,:] - coords[None,:,:]
        D = np.sqrt((diff**2).sum(-1))
        D.sort(axis=1)
        return float(D[:,1].mean())
    if KD_BACKEND == "sklearn":
        # metric='euclidean' for 2D coords
        kdt = KDTreeImpl(coords, leaf_size=40, metric='euclidean')
        def _query(): return kdt.query(coords, k=k, return_distance=True)[0]
    else:  # scipy cKDTree
        kdt = KDTreeImpl(coords)
        def _query(): return kdt.query(coords, k=k, workers=1)[0]
    if _tpl is not None:
        with _tpl(limits=max_threads):
            d = _query()
    else:
        d = _query()
    return float(np.mean(d[:,1]))

def compute_features(coords, kd_backend="auto", max_threads=1):
    # optional override
    global KD_BACKEND, KDTreeImpl
    if kd_backend in ("sklearn","scipy","numpy"):
        if kd_backend == "numpy":
            KDTreeImpl = None; KD_BACKEND = "numpy"
        elif kd_backend == "scipy":
            from scipy.spatial import cKDTree as _SPKD
            KDTreeImpl = _SPKD; KD_BACKEND = "scipy"
        else:
            from sklearn.neighbors import KDTree as _SKKD
            KDTreeImpl = _SKKD; KD_BACKEND = "sklearn"

    n = coords.shape[0]
    # normalize to improve invariance
    X = coords.astype(float)
    X -= X.mean(axis=0, keepdims=True)
    scale = np.sqrt((X**2).mean())
    if scale > 0: X /= scale

    # bbox area
    xmin, ymin = X.min(axis=0); xmax, ymax = X.max(axis=0)
    A = max((xmax - xmin) * (ymax - ymin), 1e-12)

    r1 = _mean_r1(X, k=2, max_threads=max_threads)
    delta = 2.0 * r1 * np.sqrt(n / A)              # density index

    # anisotropy via PCA (2x2 covariance)
    S = np.cov(X.T)
    evals = np.linalg.eigvalsh(S)
    lam_min = max(float(evals[0]), 1e-12)
    anisotropy = float(evals[-1] / lam_min)

    # hull fraction
    if ConvexHull is not None and n >= 3:
        try:
            hull_frac = len(ConvexHull(X).vertices) / float(n)
        except Exception:
            hull_frac = 0.0
    else:
        hull_frac = 0.0

    # crude cluster score: compare local 10-NN mean to global r1
    k_local = min(11, n)
    if KDTreeImpl is not None and k_local >= 3:
        if KD_BACKEND == "sklearn":
            kdt = KDTreeImpl(X, leaf_size=40, metric='euclidean')
            if _tpl is not None:
                with _tpl(limits=max_threads):
                    d = kdt.query(X, k=k_local, return_distance=True)[0]
            else:
                d = kdt.query(X, k=k_local, return_distance=True)[0]
        else:
            kdt = KDTreeImpl(X)
            d = kdt.query(X, k=k_local, workers=1)[0]
        local_mean = d[:,1:].mean(axis=1).mean()
    else:
        local_mean = r1
    cluster_score = float(np.clip((r1 - local_mean) / max(r1, 1e-9), 0.0, 1.0))

    return {
        "n": int(n),
        "area": float(A),
        "r1_mean": float(r1),
        "density_idx": float(delta),
        "anisotropy": float(anisotropy),
        "hull_frac": float(hull_frac),
        "cluster_score": float(cluster_score),
        "kd_backend": KD_BACKEND,
    }