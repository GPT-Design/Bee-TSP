import time
import random
import numpy as np
from typing import List, Tuple, Dict
from bee_tsp.distance import Distance

def build_initial_tours(
    n: int,
    dist: Distance,
    candidate_adj: Dict[int, List[int]],
    K: int,
    total_budget_s: float,
    mix: List[str]
) -> List[Tuple[List[int], float]]:
    """
    Build K initial tours using specified mix of heuristics.
    Returns: List of (tour, length) tuples, sorted by length (best first)
    """
    print(f"[INIT] Building {K} tours with budget {total_budget_s}s")
    start_time = time.time()
    tours = []
      
    # Map of heuristic names to functions
    heuristic_map = {
        'nn': lambda: nearest_neighbor(n, dist, candidate_adj, int(np.random.randint(0, n))),
        'nearest_ins': lambda: nearest_insertion(n, dist, candidate_adj),
        'savings': lambda: savings_algorithm(n, dist, candidate_adj),
        
    }
    
    # Build K tours, cycling through the specified heuristics
    for i in range(K):
        # Check budget
        elapsed = time.time() - start_time
        print(f"[INIT] Tour {i+1}/{K}, elapsed: {elapsed:.2f}s")
        if elapsed >= total_budget_s:
            break
        
        # Select heuristic (cycle through mix list)
        heuristic_name = mix[i % len(mix)]
        print(f"[INIT] Using heuristic: {heuristic_name}")

        # Get the heuristic function
        if heuristic_name not in heuristic_map:
            # Fallback to nearest neighbor if unknown heuristic
            heuristic_func = heuristic_map['nn']
        else:
            heuristic_func = heuristic_map[heuristic_name]
        
        # Build tour using selected heuristic
        try:
            tour = heuristic_func()
            if tour and len(tour) == n:  # Valid tour
                length = dist.tour_length(tour)
                tours.append((tour, length))
                print(f"[INIT] Built tour with length: {length:.0f}")
        except Exception as e:
            # If heuristic fails, skip it and continue
            print(f"Warning: Heuristic {heuristic_name} failed: {e}")
            continue
    
    # Sort by tour length (best first)
    tours.sort(key=lambda x: x[1])
    return tours
    
def nearest_neighbor(n, dist, candidate_adj, start):
    used = [False]*n
    used[start] = True
    tour = [start]
    while len(tour) < n:
        u = tour[-1]
        # choose the nearest unused (not the first)
        cands = [v for v in candidate_adj.get(u, []) if not used[v]]
        if cands:
            v = min(cands, key=lambda w: dist.d(u, w))
        else:
            # fall back: truly nearest over all unvisited
            v = min((j for j in range(n) if not used[j]), key=lambda j: dist.d(u, j))
        used[v] = True
        tour.append(v)
    return tour
    
def nearest_insertion(n: int, dist: Distance, candidate_adj: Dict[int, List[int]]) -> List[int]:
    print(f"[NEAREST_INS] n={n}")
    if n < 3:
        return list(range(n))
    
    # Step 1: Build initial 3-node subtour
    # Start with node 0
    tour = [0]
    used = [False] * n
    used[0] = True
    
    # Find nearest to 0
    nearest = -1
    min_dist = float('inf')
    for j in range(1, n):
        d = dist.d(0, j)
        if d < min_dist:
            min_dist = d
            nearest = j
    print(f"[NEAREST_INS] Nearest to 0 is {nearest}")
    
    tour.append(nearest)
    used[nearest] = True
    print(f"[NEAREST_INS] Tour after nearest: {tour}")
    
    # Find node farthest from both 0 and nearest
    farthest = -1
    max_min_dist = 0
    for j in range(n):
        if used[j]:
            continue
        min_d = min(dist.d(0, j), dist.d(nearest, j))
        if min_d > max_min_dist:
            max_min_dist = min_d
            farthest = j
    print(f"[NEAREST_INS] Farthest node is {farthest}, adding to tour")
    
    if farthest != -1:
        tour.append(farthest)
        used[farthest] = True
        
    print(f"[NEAREST_INS] Initial 3-node tour: {tour}, length {len(tour)}")
    
    # Step 2: Repeatedly insert nearest unvisited node
    print(f"[NEAREST_INS] Starting main insertion loop...")
    iteration = 0
    while len(tour) < n:
        iteration += 1
        if iteration % 100 == 0:
            print(f"[NEAREST_INS] Iteration {iteration}, tour length: {len(tour)}/{n}")
        
        # Find unvisited node nearest to ANY tour node    
        best_node = -1
        min_dist_to_tour = float('inf')
        
        for j in range(n):
            if used[j]:
                continue
            # Distance to nearest tour node
            min_d = min(dist.d(j, tour[i]) for i in range(len(tour)))
            if min_d < min_dist_to_tour:
                min_dist_to_tour = min_d
                best_node = j
        
        if best_node == -1:
            break
        
        # Find best insertion position (minimizes tour length increase)
        best_pos = 0
        best_increase = float('inf')
        
        for i in range(len(tour)):
            # Cost to insert best_node between tour[i] and tour[(i+1) % len(tour)]
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            increase = dist.d(a, best_node) + dist.d(best_node, b) - dist.d(a, b)
            
            if increase < best_increase:
                best_increase = increase
                best_pos = i + 1
        
        # Insert node at best position
        tour.insert(best_pos, best_node)
        used[best_node] = True
    
    return tour

def savings_algorithm(n: int, dist: Distance, candidate_adj: Dict[int, List[int]]) -> List[int]:
    """
    Clarke-Wright Savings algorithm adapted for TSP.
    
    Builds tour by greedily merging city pairs with highest savings.
    """
    if n < 2:
        return list(range(n))
    
    # Use city 0 as depot
    depot = 0
    
    # Step 1: Calculate savings for all pairs
    savings = []
    for i in range(1, n):
        for j in range(i+1, n):
            s_ij = dist.d(depot, i) + dist.d(depot, j) - dist.d(i, j)
            savings.append((s_ij, i, j))
    
    # Step 2: Sort by savings (highest first)
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Step 3: Build tour using savings
    # Track which cities are connected
    next_city = {}  # next_city[i] = j means i connects to j
    prev_city = {}  # prev_city[j] = i means j connects from i
    
    for s, i, j in savings:
        # Can only connect if:
        # - i has no next city (is an endpoint)
        # - j has no previous city (is an endpoint)
        # - They're not already in same chain
        
        if i in next_city or j in prev_city:
            continue
        
        # Check if connecting would create a loop (before all cities used)
        if i in prev_city:
            # Trace back from i
            current = i
            while current in prev_city:
                current = prev_city[current]
                if current == j:
                    # Would create premature loop
                    continue
        
        # Connect i → j
        next_city[i] = j
        prev_city[j] = i
    
    # Step 4: Build tour from connections
    # Find start (city with no predecessor)
    start = depot
    for city in range(1, n):
        if city not in prev_city:
            start = city
            break
    
    # Follow chain
    tour = [start]
    current = start
    while current in next_city:
        current = next_city[current]
        tour.append(current)
        if len(tour) >= n:
            break
    
    # Add any missing cities (shouldn't happen in valid TSP)
    visited = set(tour)
    for city in range(n):
        if city not in visited:
            tour.append(city)
    
    return tour