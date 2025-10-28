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
        'savings': lambda: savings_algorithm(n, dist, candidate_adj, randomize=True),  # ADD randomize=True
    }
    # This way:
    # - Each Savings call produces a DIFFERENT tour
    # - EAX gets diverse parents instead of identical clones
    # - Diversity = better recombination = actual improvements
    
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

def savings_algorithm(n: int, dist: Distance, candidate_adj: Dict[int, List[int]], 
                      randomize: bool = True) -> List[int]:
    """
    Clarke-Wright Savings algorithm adapted for TSP.
    
    FIXES:
    - Proper loop prevention (was broken - continue inside while loop)
    - Better handling of disconnected chains
    - Randomization for diversity
    """
    if n < 2:
        return list(range(n))
    
    # Randomize depot selection for diversity
    if randomize:
        depot = random.randint(0, n - 1)
    else:
        depot = 0
    
    # Step 1: Calculate savings for all pairs
    savings = []
    for i in range(n):
        if i == depot:
            continue
        for j in range(i+1, n):
            if j == depot:
                continue
            s_ij = dist.d(depot, i) + dist.d(depot, j) - dist.d(i, j)
            
            # Add small random noise for diversity
            if randomize:
                noise = random.uniform(-0.01, 0.01) * abs(s_ij) if s_ij != 0 else 0
                s_ij += noise
            
            savings.append((s_ij, i, j))
    
    # Step 2: Sort by savings (highest first)
    savings.sort(reverse=True, key=lambda x: x[0])
    
    # Step 3: Build tour using savings
    # Track which cities are connected
    next_city = {}  # next_city[i] = j means i connects to j
    prev_city = {}  # prev_city[j] = i means j connects from i
    
    total_pairs = len(savings)
    print(f"[SAVINGS] Evaluating {total_pairs} city pairs...")
    
    for idx, (s, i, j) in enumerate(savings):  # ← CHANGE: add enumerate
    # Progress updates
        if idx % 100000 == 0:  # ← ADD THIS (every 100K for pla33810)
            print(f"[SAVINGS] Progress: {idx:,}/{total_pairs:,} ({100*idx/total_pairs:.1f}%)")
            
        # Can only connect if:
        # - i has no next city (is an endpoint)
        # - j has no previous city (is an endpoint)
        # - They're not already in same chain
        
        if i in next_city or j in prev_city:
            continue
        
        # Check if connecting would create a loop (before all cities used)
        # FIXED: Properly skip this connection if it would create a loop
        skip_connection = False
        if i in prev_city:
            # Trace back from i to find chain start
            current = i
            while current in prev_city:
                current = prev_city[current]
                if current == j:
                    # Would create premature loop - skip this connection
                    skip_connection = True
                    break
        
        if skip_connection:
            continue  # NOW this correctly skips to next (s, i, j)
        
        # Connect i → j
        next_city[i] = j
        prev_city[j] = i
    
    # Step 4: Build tour from connections
    # Find all chain starts (nodes with no predecessor)
    chain_starts = []
    for city in range(n):
        if city not in prev_city and (city in next_city or city == depot):
            chain_starts.append(city)
    
    # If no chains formed, fall back to nearest neighbor
    if not chain_starts and not next_city:
        print(f"[SAVINGS] WARNING: No chains formed, falling back to nearest neighbor")
        return nearest_neighbor(n, dist, candidate_adj, depot)
    
    # Build main tour by following the longest chain
    best_chain = []
    for start in chain_starts:
        chain = [start]
        current = start
        while current in next_city:
            current = next_city[current]
            chain.append(current)
            if len(chain) >= n:
                break
        if len(chain) > len(best_chain):
            best_chain = chain
    
    tour = best_chain[:]
    
    # Step 5: Insert missing cities using nearest insertion
    visited = set(tour)
    missing = [city for city in range(n) if city not in visited]
    
    if missing:
        # Insert missing cities at their best positions
        for city in missing:
            # Find best insertion position (minimizes distance increase)
            best_pos = 0
            best_increase = float('inf')
            
            for i in range(len(tour)):
                a = tour[i]
                b = tour[(i + 1) % len(tour)]
                increase = dist.d(a, city) + dist.d(city, b) - dist.d(a, b)
                
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            
            # Insert city at best position
            tour.insert(best_pos, city)
    
    return tour