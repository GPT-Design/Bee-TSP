"""
Full Edge Assembly Crossover (EAX) - Nagata & Kobayashi (1997)
"""
import random
from typing import List, Dict, Set, Tuple
from collections import defaultdict, deque

def eax_full_integrate(
    pool: List[List[int]], 
    dist, 
    candidate_adj: Dict[int, List[int]], 
    config: Dict
) -> List[int]:
    """
    Full EAX integration with AB-cycle detection.
    """
    print(f"[EAX-FULL] Called with pool size: {len(pool)}")
    
    if not pool or len(pool) < 2:
        return pool[0][:] if pool else list(range(100))
    
    n = len(pool[0])
    
    # Config
    top_k = int(config.get('parents_per_round', 4))
    num_offspring = int(config.get('offspring_per_round', 2))
    repair_time_ms = int(config.get('repair_time_ms', 200))
    max_cycles_to_try = int(config.get('max_ab_cycles', 10))
    
    # Select best parents
    pool_sorted = sorted(pool, key=lambda t: tour_length(t, dist))
    parents = pool_sorted[:top_k]
    
    print(f"[EAX-FULL] Parent lengths: {[tour_length(p, dist) for p in parents[:3]]}")
    
    # Generate offspring
    best_offspring = None
    best_length = float('inf')
    
    for _ in range(num_offspring):
        # Select two random parents
        p1, p2 = random.sample(parents, 2)
        
        # Full EAX recombination
        offspring = eax_full_recombine(p1, p2, dist, candidate_adj, 
                                        repair_time_ms, max_cycles_to_try)
        
        length = tour_length(offspring, dist)
        if length < best_length:
            best_length = length
            best_offspring = offspring
    
    print(f"[EAX-FULL] Best offspring: {best_length:.0f}")
    return best_offspring

def eax_full_recombine(
    parent1: List[int],
    parent2: List[int],
    dist,
    candidate_adj: Dict[int, List[int]],
    repair_time_ms: int,
    max_cycles: int
) -> List[int]:
    """
    Full EAX with AB-cycle detection and E-set manipulation.
    """
    n = len(parent1)
    
    # Step 1: Build union graph adjacency
    # Each node has edges to its neighbors in both parents
    union_adj = build_union_graph(parent1, parent2)
    
    # Step 2: Find all AB-cycles
    ab_cycles = find_ab_cycles(parent1, parent2, union_adj, n)
    
    if not ab_cycles:
        # No cycles found - parents too similar, return better parent
        len1 = tour_length(parent1, dist)
        len2 = tour_length(parent2, dist)
        return parent1[:] if len1 < len2 else parent2[:]
    
    # Step 3: Try different cycle combinations
    best_tour = None
    best_length = float('inf')
    
    # Try random subsets of cycles
    num_attempts = min(len(ab_cycles), max_cycles)
    
    for _ in range(num_attempts):
        # Select random subset of cycles (1 to len(ab_cycles))
        k = random.randint(1, len(ab_cycles))
        selected_cycles = random.sample(ab_cycles, k)
        
        # Build E-set by applying selected cycles
        e_set = apply_cycles_to_eset(parent1, selected_cycles)
        
        # Convert E-set to valid tour
        offspring = eset_to_tour(e_set, n, dist)
        
        # Repair with 2-opt
        offspring = repair_tour_2opt(offspring, dist, candidate_adj, repair_time_ms)
        
        length = tour_length(offspring, dist)
        if length < best_length:
            best_length = length
            best_tour = offspring
    
    return best_tour if best_tour else parent1[:]


def build_union_graph(parent1: List[int], parent2: List[int]) -> Dict[int, Dict[str, Set[int]]]:
    """
    Build union graph with labeled edges.
    
    Returns: {node: {'p1': {neighbors from P1}, 'p2': {neighbors from P2}}}
    """
    n = len(parent1)
    union = defaultdict(lambda: {'p1': set(), 'p2': set()})
    
    # Add P1 edges
    for i in range(n):
        u, v = parent1[i], parent1[(i+1) % n]
        union[u]['p1'].add(v)
        union[v]['p1'].add(u)
    
    # Add P2 edges
    for i in range(n):
        u, v = parent2[i], parent2[(i+1) % n]
        union[u]['p2'].add(v)
        union[v]['p2'].add(u)
    
    return union


def find_ab_cycles(
    parent1: List[int],
    parent2: List[int],
    union_adj: Dict[int, Dict[str, Set[int]]],
    n: int
) -> List[List[Tuple[int, int, str]]]:
    """
    Find all AB-cycles (alternating P1/P2 edges).
    
    Returns: List of cycles, where each cycle is [(u, v, parent), ...]
    """
    cycles = []
    visited_nodes = set()
    
    for start_node in range(n):
        if start_node in visited_nodes:
            continue
        
        # Try to find cycle starting from this node
        cycle = find_cycle_from_node(start_node, union_adj, visited_nodes)
        
        if cycle and len(cycle) >= 4:  # Valid AB-cycle has at least 4 edges
            cycles.append(cycle)
            # Mark nodes as visited
            for u, v, _ in cycle:
                visited_nodes.add(u)
                visited_nodes.add(v)
    
    return cycles


def find_cycle_from_node(
    start: int,
    union_adj: Dict[int, Dict[str, Set[int]]],
    visited: Set[int]
) -> List[Tuple[int, int, str]]:
    """
    Find an AB-cycle starting from given node.
    
    Uses BFS to find alternating path that returns to start.
    """
    if start in visited:
        return []
    
    # Try starting with P1 edge
    cycle = []
    current = start
    last_parent = None
    path_nodes = set()
    
    while True:
        path_nodes.add(current)
        
        # Alternate between P1 and P2
        next_parent = 'p2' if last_parent == 'p1' else 'p1'
        
        # Find next node using next_parent's edges
        next_node = None
        for neighbor in union_adj[current][next_parent]:
            if neighbor == start and len(cycle) >= 2:
                # Cycle complete!
                cycle.append((current, start, next_parent))
                return cycle
            if neighbor not in path_nodes and neighbor not in visited:
                next_node = neighbor
                break
        
        if next_node is None:
            # Dead end, no cycle from this start
            return []
        
        cycle.append((current, next_node, next_parent))
        current = next_node
        last_parent = next_parent
        
        if len(cycle) > 100:  # Safety: prevent infinite loops
            return []


def apply_cycles_to_eset(
    parent1: List[int],
    cycles: List[List[Tuple[int, int, str]]]
) -> Set[Tuple[int, int]]:
    """
    Apply AB-cycles to parent1 tour to create E-set.
    
    E-set starts as P1's edges, then swaps edges for selected cycles.
    """
    n = len(parent1)
    
    # Start with all P1 edges
    eset = set()
    for i in range(n):
        u, v = parent1[i], parent1[(i+1) % n]
        edge = (min(u, v), max(u, v))
        eset.add(edge)
    
    # For each cycle, swap P1 edges with P2 edges
    for cycle in cycles:
        for u, v, parent in cycle:
            edge = (min(u, v), max(u, v))
            if parent == 'p1':
                # Remove P1 edge
                eset.discard(edge)
            else:
                # Add P2 edge
                eset.add(edge)
    
    return eset


def eset_to_tour(eset: Set[Tuple[int, int]], n: int, dist) -> List[int]:
    """
    Convert E-set (set of edges) to a valid Hamiltonian tour.
    
    E-set might form one tour or multiple subtours.
    If multiple, connect them with shortest links.
    """
    # Build adjacency from E-set
    adj = defaultdict(list)
    for u, v in eset:
        adj[u].append(v)
        adj[v].append(u)
    
    # Find all connected components (subtours)
    visited = [False] * n
    subtours = []
    
    for start in range(n):
        if visited[start]:
            continue
        
        # BFS to find connected component
        subtour = []
        queue = deque([start])
        visited[start] = True
        
        while queue:
            node = queue.popleft()
            subtour.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        if subtour:
            subtours.append(subtour)
    
    # If single tour, convert to proper order
    if len(subtours) == 1:
        return order_subtour(subtours[0], adj)
    
    # Multiple subtours - merge them
    return merge_subtours(subtours, dist, n)


def order_subtour(nodes: List[int], adj: Dict[int, List[int]]) -> List[int]:
    """
    Order nodes in subtour to form a path/tour.
    """
    if len(nodes) <= 2:
        return nodes
    
    # Start from first node, follow edges
    tour = [nodes[0]]
    visited = {nodes[0]}
    current = nodes[0]
    
    while len(tour) < len(nodes):
        # Find unvisited neighbor
        next_node = None
        for neighbor in adj[current]:
            if neighbor not in visited and neighbor in nodes:
                next_node = neighbor
                break
        
        if next_node is None:
            break
        
        tour.append(next_node)
        visited.add(next_node)
        current = next_node
    
    return tour


def merge_subtours(subtours: List[List[int]], dist, n: int) -> List[int]:
    """
    Merge multiple subtours into single tour by connecting with shortest links.
    """
    if not subtours:
        return list(range(n))
    
    if len(subtours) == 1:
        return subtours[0]
    
    # Start with first subtour
    tour = subtours[0][:]
    remaining = subtours[1:]
    
    while remaining:
        # Find closest subtour to current tour
        min_dist = float('inf')
        best_sub_idx = 0
        best_insert_pos = 0
        
        for sub_idx, subtour in enumerate(remaining):
            # Try connecting at each position in current tour
            for insert_pos in range(len(tour)):
                # Distance from tour[insert_pos] to subtour[0]
                d = dist.d(tour[insert_pos], subtour[0])
                if d < min_dist:
                    min_dist = d
                    best_sub_idx = sub_idx
                    best_insert_pos = insert_pos + 1
        
        # Insert best subtour
        subtour = remaining.pop(best_sub_idx)
        tour = tour[:best_insert_pos] + subtour + tour[best_insert_pos:]
    
    return tour


def repair_tour_2opt(tour: List[int], dist, candidate_adj: Dict[int, List[int]], 
                     time_budget_ms: int) -> List[int]:
    """2-opt repair (same as before)"""
    import time
    
    if not tour or len(tour) < 4:
        return tour
    
    start_time = time.time() * 1000
    n = len(tour)
    tour = tour[:]
    
    pos = [0] * n
    for i, city in enumerate(tour):
        pos[city] = i
    
    improved = True
    while improved and (time.time() * 1000 - start_time) < time_budget_ms:
        improved = False
        
        for i in range(n):
            if (time.time() * 1000 - start_time) >= time_budget_ms:
                break
            
            for city_k in candidate_adj.get(tour[i], [])[:5]:
                k = pos[city_k]
                if k == i or k == (i+1) % n or k == (i-1) % n:
                    continue
                
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[k], tour[(k+1) % n]
                delta = (dist.d(a, c) + dist.d(b, d)) - (dist.d(a, b) + dist.d(c, d))
                
                if delta < -0.1:
                    if i < k:
                        tour[i+1:k+1] = tour[i+1:k+1][::-1]
                    else:
                        tour[k+1:i+1] = tour[k+1:i+1][::-1]
                    
                    for idx, city in enumerate(tour):
                        pos[city] = idx
                    
                    improved = True
                    break
            
            if improved:
                break
    
    return tour


def tour_length(tour: List[int], dist) -> float:
    """Calculate tour length"""
    if not tour:
        return float('inf')
    n = len(tour)
    return sum(dist.d(tour[i], tour[(i+1) % n]) for i in range(n))