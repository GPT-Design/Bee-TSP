#!/usr/bin/env python3
"""
Long Wall Batch Monitor
Monitors progress of long-running batch jobs
"""

import time
import json
import psutil
from pathlib import Path

def monitor_batch():
    """Monitor batch progress and system resources."""
    
    print("=== LONG WALL BATCH MONITOR ===")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"\r[{time.strftime('%H:%M:%S')}] CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% | Available: {memory.available/1024**3:.1f}GB", end='', flush=True)
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_batch()
