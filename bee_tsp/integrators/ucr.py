# bee_tsp/integrators/ucr.py
"""
Union-Cycle Recombination (UCR) - LITE version of EAX-like recombination
"""
import random
import time
from typing import List, Dict, Any, Tuple, Set


def tour_length(tour: List[int], dist) -> float:
    """Calculate tour length."""
    if not tour:
        return float('inf')
    n = len(tour)
    return sum(dist(tour[i], tour[(i+1) % n]) for i in range(n))


def find_symmetric_difference_cycles(edges_a: Set[Tuple[int, int]], edges_b: Set[Tuple[int, int]]) -> List[List[int]]:
    """Find AB-cycles (symmetric difference cycles) between two tours."""
    # Symmetric difference: edges in A XOR B
    sym_diff = edges_a.symmetric_difference(edges_b)
    
    if not sym_diff:
        return []
    
    # Build adjacency from symmetric difference
    adj = {}
    for u, v in sym_diff:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    
    cycles = []
    visited = set()
    
    for start in adj:
        if start in visited:
            continue
            
        # Follow the cycle
        cycle = []
        current = start
        prev = None
        
        while current not in visited:
            visited.add(current)
            cycle.append(current)
            
            # Find next node (degree should be 2 in symmetric difference)
            neighbors = [n for n in adj[current] if n != prev]
            if not neighbors:
                break
                
            prev = current
            current = neighbors[0]
            
            # Stop if we've returned to start
            if current == start:
                break
        
        if len(cycle) > 2:  # Valid cycle
            cycles.append(cycle)
    
    return cycles


def get_tour_edges(tour: List[int]) -> Set[Tuple[int, int]]:
    """Extract edges from tour as set of (min, max) tuples."""
    if not tour:
        return set()
    
    edges = set()
    n = len(tour)
    for i in range(n):
        u, v = tour[i], tour[(i+1) % n]
        edges.add((min(u, v), max(u, v)))
    return edges


def apply_bounded_2opt(tour: List[int], dist, candidate_adj: Dict[int, List[int]], time_budget_ms: int) -> List[int]:
    """Apply bounded 2-opt repair with time constraint."""
    if not tour or len(tour) < 4:
        return tour
    
    start_time = time.time() * 1000  # Convert to ms
    n = len(tour)
    tour = tour[:]  # Make copy
    
    # Build position map
    pos = [0] * n
    for i, city in enumerate(tour):
        pos[city] = i
    
    improved = True
    while improved and (time.time() * 1000 - start_time) < time_budget_ms:
        improved = False
        
        for i in range(n):
            if (time.time() * 1000 - start_time) >= time_budget_ms:
                break
                
            city_i = tour[i]
            # Check candidates from candidate graph
            for city_k in candidate_adj.get(city_i, [])[:5]:  # Limit to top 5
                k = pos[city_k]
                if k == i or k == (i + 1) % n or k == (i - 1) % n:
                    continue
                
                # Calculate 2-opt delta
                a, b = tour[i], tour[(i+1) % n]
                c, d = tour[k], tour[(k+1) % n]
                delta = (dist(a, c) + dist(b, d)) - (dist(a, b) + dist(c, d))
                
                if delta < -0.1:  # Improvement threshold
                    # Apply 2-opt move
                    if i < k:
                        tour[i+1:k+1] = tour[i+1:k+1][::-1]
                    else:
                        tour[k+1:i+1] = tour[k+1:i+1][::-1]
                    
                    # Update position map
                    for idx, city in enumerate(tour):
                        pos[city] = idx
                    
                    improved = True
                    break
            
            if improved:
                break
    
    return tour


def ucr_recombine_pair(parent_a: List[int], parent_b: List[int], dist, candidate_adj: Dict[int, List[int]], config: Dict[str, Any]) -> List[int]:
    """
    Union-Cycle Recombination between two parents.
    """
    if not parent_a or not parent_b or len(parent_a) != len(parent_b):
        return parent_a[:]  # Return copy of first parent as fallback
    
    repair_budget_ms = int(config.get('repair_budget_ms', 50))
    
    # Get edge sets from both parents
    edges_a = get_tour_edges(parent_a)
    edges_b = get_tour_edges(parent_b)
    
    # Find AB-cycles (symmetric difference cycles)
    cycles = find_symmetric_difference_cycles(edges_a, edges_b)
    
    if not cycles:
        # No symmetric difference, return one of the parents
        return parent_a[:]
    
    # Create child by randomly flipping some cycles
    # Start with parent A edges
    child_edges = edges_a.copy()
    
    # For each cycle, randomly decide whether to flip it
    for cycle in cycles:
        if random.random() < 0.5:  # 50% chance to flip
            # Remove edges from A that are in this cycle
            cycle_edges_a = set()
            cycle_edges_b = set()
            
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                
                if edge in edges_a:
                    cycle_edges_a.add(edge)
                if edge in edges_b:
                    cycle_edges_b.add(edge)
            
            # Remove A edges and add B edges for this cycle
            child_edges -= cycle_edges_a
            child_edges |= cycle_edges_b
    
    # Construct tour from child edges (this might create subtours)
    child_tour = construct_tour_from_edges(child_edges, len(parent_a))
    
    # Apply bounded 2-opt repair
    if candidate_adj:
        child_tour = apply_bounded_2opt(child_tour, dist, candidate_adj, repair_budget_ms)
    
    return child_tour


def construct_tour_from_edges(edges: Set[Tuple[int, int]], n: int) -> List[int]:
    """
    Construct a tour from a set of edges. 
    If edges don't form a single cycle, use nearest neighbor to complete.
    """
    if not edges:
        return list(range(n))
    
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find a single cycle starting from node 0
    tour = []
    visited = set()
    current = 0
    
    while len(tour) < n:
        tour.append(current)
        visited.add(current)
        
        # Find next unvisited neighbor
        next_node = None
        for neighbor in adj[current]:
            if neighbor not in visited:
                next_node = neighbor
                break
        
        if next_node is None:
            # No unvisited neighbors, find nearest unvisited node
            min_dist = float('inf')
            for i in range(n):
                if i not in visited:
                    # Use simple Euclidean approximation for nearest neighbor
                    dist = abs(i - current)  # Simplified distance
                    if dist < min_dist:
                        min_dist = dist
                        next_node = i
        
        if next_node is None:
            # All nodes visited
            break
            
        current = next_node
    
    # Ensure we have all nodes
    for i in range(n):
        if i not in visited:
            tour.append(i)
    
    return tour[:n]  # Truncate to exactly n nodes


def ucr_integrate(pool: List[List[int]], dist, candidate_adj: Dict[int, List[int]], config: Dict[str, Any]) -> List[int]:
    """
    UCR integration: recombine multiple parents and return best child.
    """
    if not pool:
        return list(range(10))  # Fallback
    
    if len(pool) == 1:
        return pool[0][:]
    
    parents = int(config.get('parents', 6))
    children_per_pair = int(config.get('children_per_pair', 1))
    
    # Select best parents
    pool_with_lengths = [(tour, tour_length(tour, dist)) for tour in pool]
    pool_with_lengths.sort(key=lambda x: x[1])
    
    selected_parents = [tour for tour, _ in pool_with_lengths[:parents]]
    
    if len(selected_parents) < 2:
        return selected_parents[0][:]
    
    # Generate children through pairwise recombination
    children = []
    
    for i in range(len(selected_parents)):
        for j in range(i + 1, len(selected_parents)):
            parent_a, parent_b = selected_parents[i], selected_parents[j]
            
            for _ in range(children_per_pair):
                child = ucr_recombine_pair(parent_a, parent_b, dist, candidate_adj, config)
                children.append(child)
    
    # Return best child
    if children:
        best_child = min(children, key=lambda tour: tour_length(tour, dist))
        return best_child
    else:
        return selected_parents[0][:]