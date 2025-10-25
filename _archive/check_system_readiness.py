#!/usr/bin/env python3
"""
System Resource Checker for Long Wall Batch
"""

import psutil
import os

def check_system_readiness():
    """Check if system is ready for long wall batch processing."""
    
    print("=== SYSTEM READINESS CHECK ===")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    print(f"CPU: {cpu_count} cores @ {cpu_freq.current:.0f}MHz")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Memory: {memory.total/1024**3:.1f}GB total, {memory.available/1024**3:.1f}GB available ({100-memory.percent:.1f}%)")
    
    # Disk space
    disk = psutil.disk_usage('.')
    print(f"Disk: {disk.free/1024**3:.1f}GB free of {disk.total/1024**3:.1f}GB ({disk.percent:.1f}% used)")
    
    # Environment check
    print("\nEnvironment Variables:")
    env_vars = ['STRICT_TSPLIB_EVAL', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        status = "OK" if value == '1' else "WARN"
        print(f"  {status} {var}: {value}")
    
    # Recommendations
    print("\nRecommendations:")
    if memory.available < 4 * 1024**3:  # Less than 4GB
        print("  WARN Low available memory - consider closing other applications")
    if disk.free < 10 * 1024**3:  # Less than 10GB
        print("  WARN Low disk space - ensure adequate space for logs and results")
    if memory.available >= 8 * 1024**3 and disk.free >= 50 * 1024**3:
        print("  OK System appears ready for long wall batch processing")
    
    return memory.available >= 2 * 1024**3 and disk.free >= 5 * 1024**3

if __name__ == "__main__":
    ready = check_system_readiness()
    exit(0 if ready else 1)
