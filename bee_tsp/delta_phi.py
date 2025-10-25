"""
ΔΦ (Delta-Phi) edge quality metric from ENC framework.
"""
import math
from typing import Dict, Set, Tuple, List

def compute_delta_phi_for_edges(
    n: int,
    dist,
    candidate_adj: Dict[int, List[int]],
    config: Dict
) -> Dict[Tuple[int, int], float]:
    """
    Compute ΔΦ for all edges in candidate graph.
    """
    print("[ΔΦ] Computing edge quality metrics...")
    
    lambda_trade = float(config.get('lambda_trade', 0.1))
    
    delta_phi = {}
    edge_count = 0
    
    for u in range(n):
        for v in candidate_adj.get(u, []):
            if u >= v:
                continue
            
            edge = (u, v)
            
            # Pass config to structural_benefit
            structural_benefit = compute_structural_benefit(
                u, v, n, dist, candidate_adj, config  # ← PASS CONFIG
            )
            
            entropy_cost = compute_entropy_cost(u, v, candidate_adj)
            
            dphi = structural_benefit - lambda_trade * entropy_cost
            
            delta_phi[edge] = dphi
            edge_count += 1
    
    print(f"[ΔΦ] Computed ΔΦ for {edge_count} edges")
    
    positive = sum(1 for dp in delta_phi.values() if dp > 0)
    negative = sum(1 for dp in delta_phi.values() if dp <= 0)
    avg_dphi = sum(delta_phi.values()) / len(delta_phi) if delta_phi else 0
    
    print(f"[ΔΦ] Positive: {positive}, Negative: {negative}, Avg: {avg_dphi:.3f}")
    
    return delta_phi

def compute_structural_benefit(
    u: int, 
    v: int, 
    n: int, 
    dist, 
    candidate_adj: Dict[int, List[int]],
    config: Dict  # ← ADD THIS PARAMETER
) -> float:
    """
    Compute Δ(E_i R_c) - structural benefit of edge (u,v).
    Now uses adaptive config for weighting.
    """
    d_uv = dist.d(u, v)
    
    # Efficiency component: short edges are beneficial
    efficiency_scale = config.get('efficiency_scale', 1000.0)
    efficiency = efficiency_scale / (d_uv + 1.0)
    
    # Reliability component: edges that bridge or connect hubs
    neighbors_u = set(candidate_adj.get(u, []))
    neighbors_v = set(candidate_adj.get(v, []))
    common_neighbors = len(neighbors_u & neighbors_v)
    
    max_common = min(len(neighbors_u), len(neighbors_v))
    if max_common > 0:
        hub_score = common_neighbors / max_common  # High = hub connection
    else:
        hub_score = 0.5
    
    bridge_score = 1.0 - hub_score  # High = bridging edge
    
    # Weighted combination based on adaptive config
    hub_weight = config.get('hub_weight', 1.0)
    bridge_weight = config.get('bridge_weight', 1.0)
    
    structural_benefit = (
        efficiency + 
        hub_weight * hub_score + 
        bridge_weight * bridge_score
    )
    
    return structural_benefit

def compute_entropy_cost(
    u: int,
    v: int,
    candidate_adj: Dict[int, List[int]]
) -> float:
    """
    Compute Δh - entropy cost of edge (u,v).
    
    Approximation:
    Decision entropy = log(branching factor)
    
    Edge (u,v) increases entropy if it adds to high-degree nodes.
    Cost = (degree(u) + degree(v)) / 2
    """
    degree_u = len(candidate_adj.get(u, []))
    degree_v = len(candidate_adj.get(v, []))
    
    # Entropy cost: average degree
    # High-degree nodes already have high entropy
    avg_degree = (degree_u + degree_v) / 2.0
    
    # Normalize by log (entropy is logarithmic)
    entropy_cost = math.log(avg_degree + 1.0)
    
    return entropy_cost

def filter_candidate_graph_by_delta_phi(
    candidate_adj: Dict[int, List[int]],
    delta_phi: Dict[Tuple[int, int], float],
    n: int,
    dist,
    config: Dict
) -> Dict[int, List[int]]:
    """
    Prune candidate graph using PERCENTILE-BASED filtering.
    """
    print("[ΔΦ] Filtering candidate graph (percentile-based)...")
    
    # Get pruning percentage from config
    prune_pct = float(config.get('prune_bottom_pct', 30.0))  # Prune bottom 30%
    min_degree = int(config.get('min_degree_after_pruning', 5))
    
    # Get all ΔΦ values and sort
    all_dphi_values = list(delta_phi.values())
    all_dphi_values.sort()
    
    # Find threshold at given percentile
    threshold_idx = int(len(all_dphi_values) * (prune_pct / 100.0))
    threshold = all_dphi_values[threshold_idx]
    
    print(f"[ΔΦ] Pruning bottom {prune_pct}% of edges (threshold: {threshold:.3f})")
    
    # Filter using this threshold
    filtered_adj = {i: [] for i in range(n)}
    pruned_count = 0
    kept_count = 0
    
    for u in range(n):
        for v in candidate_adj.get(u, []):
            edge = (min(u, v), max(u, v))
            dphi = delta_phi.get(edge, 0.0)
            
            if dphi > threshold:  # Above threshold = keep
                filtered_adj[u].append(v)
                if u < v:
                    kept_count += 1
            else:
                if u < v:
                    pruned_count += 1
    
    # Ensure minimum degree (safety net)
    for u in range(n):
        if len(filtered_adj[u]) < min_degree:
            original = candidate_adj.get(u, [])
            needed = min_degree - len(filtered_adj[u])
            for v in original[:needed]:
                if v not in filtered_adj[u]:
                    filtered_adj[u].append(v)
    
    # Sort by distance
    for u in range(n):
        filtered_adj[u] = sorted(filtered_adj[u], key=lambda v: dist.d(u, v))
    
    print(f"[ΔΦ] Pruned {pruned_count} edges, kept {kept_count} edges")
    
    return filtered_adj

def add_high_delta_phi_edges(
    candidate_adj: Dict[int, List[int]],
    delta_phi: Dict[Tuple[int, int], float],
    n: int,
    dist,
    config: Dict
) -> Dict[int, List[int]]:
    """
    Add high-ΔΦ edges that aren't in candidate graph.
    
    Strategy: For each node, add top-k highest-ΔΦ edges.
    """
    print("[ΔΦ] Adding high-ΔΦ edges...")
    
    max_additions_per_node = int(config.get('max_delta_phi_additions', 3))
    min_dphi_to_add = float(config.get('min_dphi_to_add', 0.5))
    
    enhanced_adj = {i: list(candidate_adj.get(i, [])) for i in range(n)}
    added_count = 0
    
    # For each node, find high-ΔΦ edges not currently in candidate graph
    for u in range(n):
        current_neighbors = set(enhanced_adj[u])
        
        # Find edges involving u with high ΔΦ
        candidates = []
        for v in range(n):
            if v == u or v in current_neighbors:
                continue
            
            edge = (min(u, v), max(u, v))
            dphi = delta_phi.get(edge, 0.0)
            
            if dphi >= min_dphi_to_add:
                candidates.append((dphi, v))
        
        # Add top-k
        candidates.sort(reverse=True)
        for dphi, v in candidates[:max_additions_per_node]:
            enhanced_adj[u].append(v)
            added_count += 1
    
    # Re-sort by distance
    for u in range(n):
        enhanced_adj[u] = sorted(enhanced_adj[u], key=lambda v: dist.d(u, v))
    
    print(f"[ΔΦ] Added {added_count} high-ΔΦ edges")
    
    return enhanced_adj