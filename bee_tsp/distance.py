#!/usr/bin/env python3
"""
Unified Distance Factory for TSP Solver
Step 2 of metric drift fix - Replace all ad-hoc distance calls
"""
import math
from typing import List, Tuple, Callable

Coord = Tuple[float, float]

class Distance:
    """Unified distance calculator that ensures consistent metric usage throughout the solver."""

    def __init__(self, metric_tag: str, coords: List[Coord]):
        self.metric = metric_tag.upper()
        self.coords = coords

        # Map metric to implementation
        metric_map = {
            "EUC_2D": self._euc,
            "CEIL_2D": self._ceil,
            "ATT": self._att,
            "GEO": self._geo,
            "MAN_2D": self._manhattan,
            "MAX_2D": self._maximum
        }

        if self.metric not in metric_map:
            raise ValueError(f"Unsupported metric: {self.metric}")

        # Precompute radians for GEO metric
        if self.metric == "GEO":
            self._precompute_geo_radians()
        
        self._f = metric_map[self.metric]

        print(f"[DISTANCE_FACTORY] Initialized with metric={self.metric}, n_coords={len(coords)}")

    def _precompute_geo_radians(self):
        """Precompute radians for GEO metric using TSPLIB formula."""
        self._geo_rad = [(self._geo_to_radians(x), self._geo_to_radians(y)) 
                         for x, y in self.coords]
            # TSPLIB GEO: coordinates are DDD.MM (degrees.minutes)
            #lat_rad = self._geo_to_radians(x)
            #lon_rad = self._geo_to_radians(y)
            #self._geo_rad.append((lat_rad, lon_rad))
    
    def _geo_to_radians(self, val: float) -> float:
        """
        Convert TSPLIB GEO coordinate to radians.
        
        TSPLIB GEO format: DDD.MM where .MM are MINUTES (not decimal degrees!)
        Example: 71.17 = 71 degrees + 17 minutes
        
        TSPLIB formula: π × (deg + 5.0 × minutes / 3.0) / 180.0
        """
        deg = int(val)
        minutes = val - deg
        return math.pi * (deg + 5.0 * minutes / 3.0) / 180.0 

    def _euc(self, i: int, j: int) -> int:
        """Euclidean distance (integer, rounded)"""
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        return int(round(math.hypot(x1 - x2, y1 - y2)))

    def _ceil(self, i: int, j: int) -> int:
        """Euclidean distance (integer, ceiling)"""
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        return int(math.ceil(math.hypot(x1 - x2, y1 - y2)))

    def _att(self, i: int, j: int) -> int:
        """ATT (pseudo-Euclidean) distance"""
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        rij = math.sqrt(((x1 - x2)**2 + (y1 - y2)**2) / 10.0)
        t = int(round(rij))
        return t if t >= rij else t + 1

    def _geo(self, i: int, j: int) -> int:
        """
        Geographic distance using TSPLIB specification.
        
        CRITICAL: Must match TSPLIB exactly!
        - Uses precomputed radians from _geo_to_radians()
        - R = 6378.388 (TSPLIB value, NOT 6371.0)
        - Uses TSPLIB's specific formula (NOT Haversine!)
        """
        lat_i, lon_i = self._geo_rad[i]
        lat_j, lon_j = self._geo_rad[j]
        
        # TSPLIB GEO formula
        R = 6378.388  # Earth radius in km (TSPLIB value)
        
        q1 = math.cos(lon_i - lon_j)
        q2 = math.cos(lat_i - lat_j)
        q3 = math.cos(lat_i + lat_j)
        
        dij = R * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
        
        return int(dij)  # Truncate (TSPLIB rule)

    def _manhattan(self, i: int, j: int) -> int:
        """Manhattan (L1) distance"""
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        return int(round(abs(x1 - x2) + abs(y1 - y2)))

    def _maximum(self, i: int, j: int) -> int:
        """Maximum (L∞) distance"""
        x1, y1 = self.coords[i]
        x2, y2 = self.coords[j]
        return int(round(max(abs(x1 - x2), abs(y1 - y2))))

    def d(self, i: int, j: int) -> int:
        """Main distance function - use this everywhere instead of ad-hoc calculations."""
        return self._f(i, j)

    def __call__(self, i: int, j: int) -> int:
        """Allow calling as dist(i, j) for compatibility"""
        return self.d(i, j)

    def get_metric(self) -> str:
        """Return the current metric type"""
        return self.metric

    def tour_length(self, tour: List[int]) -> int:
        """Calculate total tour length using consistent metric"""
        n = len(tour)
        return sum(self.d(tour[i], tour[(i + 1) % n]) for i in range(n))

    def dual_metric_sentinel(self, tour: List[int], length_tsplib: int) -> None:
        """
        CEIL/EUC dual re-sum sentinel - catches metric drift bugs.
        For final tour, compute L_EUC, L_CEIL from same coords.
        If metric_tag=='EUC_2D' and |L_EUC - length_tsplib| > 0.5% -> FAIL
        If metric_tag=='EUC_2D' and |L_CEIL - length_tsplib| <= 0.1% -> FAIL (caught bug)
        """
        if self.metric != "EUC_2D":
            return  # Only check for EUC_2D instances

        # Create both distance calculators for comparison
        euc_dist = Distance("EUC_2D", self.coords)
        ceil_dist = Distance("CEIL_2D", self.coords)

        # Calculate tour lengths with both metrics
        l_euc = euc_dist.tour_length(tour)
        l_ceil = ceil_dist.tour_length(tour)

        # Check 1: EUC_2D should match within 0.5%
        euc_error_pct = abs(l_euc - length_tsplib) / max(length_tsplib, 1) * 100
        if euc_error_pct > 0.5:
            raise AssertionError(f"METRIC DRIFT: EUC_2D length {l_euc} differs from reported {length_tsplib} by {euc_error_pct:.2f}% > 0.5%")

        # Check 2: CEIL_2D should NOT match closely (if it does, we have a bug)
        ceil_error_pct = abs(l_ceil - length_tsplib) / max(length_tsplib, 1) * 100
        if ceil_error_pct <= 0.1:
            raise AssertionError(f"METRIC DRIFT BUG: CEIL_2D length {l_ceil} matches reported {length_tsplib} within {ceil_error_pct:.2f}% <= 0.1% - solver using wrong metric!")

        print(f"[SENTINEL] EUC_2D={l_euc}, CEIL_2D={l_ceil}, reported={length_tsplib}, EUC_err={euc_error_pct:.2f}%, CEIL_err={ceil_error_pct:.2f}%")