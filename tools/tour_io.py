# tools/tour_io.py
import hashlib, os, json

def is_perm_0n(tour, n):
    """True if tour is a 0..n-1 permutation."""
    if not isinstance(n, int) or n <= 0 or not isinstance(tour, (list, tuple)) or len(tour) != n:
        return False
    try:
        # fast bounds + uniqueness check
        s = set(tour)
        if len(s) != n:
            return False
        if min(s) != 0 or max(s) != n - 1:
            return False
        # guard non-ints / out-of-range silently sneaking in
        for v in tour:
            if not isinstance(v, int) or v < 0 or v >= n:
                return False
        return True
    except ValueError:
        return False

def write_tsplib_tour(path, name, tour, length, metric, dim=None):
    """
    Write TSPLIB TOUR with 1-based IDs. Ensures parent dir exists.
    Returns the path written.
    """
    n = len(tour)
    dim = int(dim) if dim is not None else n

    # sanity before write
    if dim != n:
        # prefer self-consistency over header mismatch
        dim = n
    if not is_perm_0n(tour, n):
        raise ValueError("[TSPLIB] Tour is not a valid 0..n-1 permutation")

    # make parent dir
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    # write file (1-based IDs)
    lines = [
        f"NAME : {name}",
        "TYPE : TOUR",
        f"DIMENSION : {dim}",
        f"COMMENT : Bee-TSP Mark1; metric={metric}; length={int(round(float(length)))}",
        "TOUR_SECTION",
        *(str(int(c) + 1) for c in tour),
        "-1",
        "EOF",
        ""
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return path

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def append_manifest_row(manifest_path, row: dict):
    """Append one JSON object per line to manifest; ensures parent dir exists."""
    d = os.path.dirname(manifest_path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
