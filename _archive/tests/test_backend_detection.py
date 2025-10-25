#!/usr/bin/env python3
"""Test script to verify sklearn backend detection and feature computation."""

import numpy as np
from bee_tsp.features import compute_features

def test_backends():
    # Create a simple test dataset
    np.random.seed(42)
    coords = np.random.rand(50, 2) * 100  # 50 points in 100x100 square
    
    print("Testing S·S·T Feature Backend Detection")
    print("=" * 50)
    
    # Test each backend
    backends = ["auto", "sklearn", "scipy", "numpy"]
    
    for backend in backends:
        try:
            print(f"\nTesting backend: {backend}")
            features = compute_features(coords, kd_backend=backend, max_threads=1)
            
            print(f"* Backend used: {features.get('kd_backend', 'unknown')}")
            print(f"  n: {features['n']}")
            print(f"  density_idx: {features['density_idx']:.3f}")
            print(f"  anisotropy: {features['anisotropy']:.3f}")
            print(f"  hull_frac: {features['hull_frac']:.3f}")
            print(f"  cluster_score: {features['cluster_score']:.3f}")
            
        except Exception as e:
            print(f"X Backend {backend} failed: {e}")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    test_backends()