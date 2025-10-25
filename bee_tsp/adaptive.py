"""
Adaptive parameter selection based on instance structure analysis.
"""
import numpy as np
from typing import Dict, List, Tuple

def analyze_instance_structure(coords: List[Tuple[float, float]], dist, n: int) -> Dict:
    """
    Detect instance characteristics to guide parameter selection.
    
    Returns metrics:
    - clustering_coefficient: How clustered is the instance?
    - hub_score: How prominent are hub nodes?
    - size: Number of cities
    """
    print("[ADAPTIVE] Analyzing instance structure...")
    
    # Build simple k-NN graph for analysis (k=20)
    k = min(20, max(5, n // 100))  # Scale k with instance size
    knn_graph = {}
    
    print(f"[ADAPTIVE] Building k-NN graph with k={k}...")
    
    # For each city, find k nearest neighbors
    for i in range(n):
        # Compute distances to all other cities
        distances = []
        for j in range(n):
            if i != j:
                distances.append((dist.d(i, j), j))
        
        # Sort by distance and take k nearest
        distances.sort()
        knn_graph[i] = [j for _, j in distances[:k]]
    
    # Metric 1: Clustering coefficient
    clustering = compute_clustering_coefficient(knn_graph, n)
    
    # Metric 2: Hub prominence
    degrees = [len(knn_graph[i]) for i in range(n)]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    hub_score = max_degree / avg_degree if avg_degree > 0 else 1.0
    
    metrics = {
        'clustering': clustering,
        'hub_score': hub_score,
        'n': n,
        'avg_degree': avg_degree,
        'max_degree': max_degree
    }
    
    print(f"[ADAPTIVE] Clustering coefficient: {clustering:.3f}")
    print(f"[ADAPTIVE] Hub score: {hub_score:.2f} (max: {max_degree}, avg: {avg_degree:.1f})")
    print(f"[ADAPTIVE] Instance size: {n} cities")
    
    return metrics


def compute_clustering_coefficient(adj: Dict[int, List[int]], n: int) -> float:
    """
    Clustering coefficient: how many of your neighbors are also neighbors?
    High = clustered instance, Low = random instance
    """
    total_cc = 0.0
    valid_nodes = 0
    
    for u in range(n):
        neighbors = set(adj.get(u, []))
        if len(neighbors) < 2:
            continue
        
        valid_nodes += 1
        
        # Count edges between neighbors
        edges_between = 0
        for v in neighbors:
            v_neighbors = set(adj.get(v, []))
            # Count common neighbors
            common = neighbors & v_neighbors
            edges_between += len(common)
        
        # Each edge counted twice, divide by 2
        edges_between = edges_between / 2.0
        
        # Possible edges between neighbors
        possible = len(neighbors) * (len(neighbors) - 1) / 2.0
        if possible > 0:
            total_cc += edges_between / possible
    
    return total_cc / valid_nodes if valid_nodes > 0 else 0.0


def adaptive_delta_phi_config(structure_metrics: Dict) -> Dict:
    """
    Choose ΔΦ parameters based on instance structure.
    
    Returns config dict for ΔΦ computation.
    """
    clustering = structure_metrics['clustering']
    hub_score = structure_metrics['hub_score']
    n = structure_metrics['n']
    
    print(f"[ADAPTIVE] Selecting ΔΦ strategy...")
    
    # RULE 1: Highly clustered with strong hubs → Hub-preserving mode
    if clustering > 0.4 and hub_score > 2.0:
        print(f"[ADAPTIVE] → CLUSTERED with HUBS (cluster={clustering:.2f}, hubs={hub_score:.2f})")
        print(f"[ADAPTIVE] → Strategy: Preserve hub connections, minimal pruning")
        return {
            'strategy': 'hub_preserving',
            'lambda_trade': 0.005,  # Very low penalty
            'prune_bottom_pct': 15.0,  # Minimal pruning
            'efficiency_scale': 1000.0,
            'hub_weight': 3.0,  # STRONGLY favor hub edges
            'bridge_weight': 0.0
        }
    
    # RULE 2: Low clustering, even degree → Bridge-seeking mode
    elif clustering < 0.25 and hub_score < 1.5:
        print(f"[ADAPTIVE] → RANDOM/UNIFORM (cluster={clustering:.2f}, hubs={hub_score:.2f})")
        print(f"[ADAPTIVE] → Strategy: Prioritize bridging, aggressive pruning")
        return {
            'strategy': 'bridge_seeking',
            'lambda_trade': 0.1,
            'prune_bottom_pct': 40.0,
            'efficiency_scale': 1000.0,
            'hub_weight': 0.0,
            'bridge_weight': 2.0
        }
    
    # RULE 3: Mixed structure → Balanced mode
    else:
        print(f"[ADAPTIVE] → MIXED structure (cluster={clustering:.2f}, hubs={hub_score:.2f})")
        print(f"[ADAPTIVE] → Strategy: Balanced approach")
        return {
            'strategy': 'balanced',
            'lambda_trade': 0.05,
            'prune_bottom_pct': 25.0,
            'efficiency_scale': 1000.0,
            'hub_weight': 1.0,
            'bridge_weight': 1.0
        }


def size_adaptive_params(n: int) -> Dict:
    """
    Adjust solver parameters based on instance size.
    """
    if n < 1000:
        return {'candidate_k': 30}
    elif n < 5000:
        return {'candidate_k': 40}
    else:
        return {'candidate_k': 50}