"""
Route Finding Algorithm Demo File
Running BFS and A* Algorithms and Comparing Results
"""

import sys
import os
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_finding_comparison import compare_algorithms

def main():
    """
    Main function - Run route finding algorithm demo
    """
    print("ITP4514 - Route Finding Algorithm Demo")
    print("Author: Li Xueqing")
    print("Date: November 2025")
    print("=" * 50)
    print("Demo Content: Breadth-First Search (BFS) vs A* Search Algorithm")
    print("=" * 50)
    
    try:
        compare_algorithms()
    except Exception as e:
        print(f"Error running route finding algorithm demo: {e}")

if __name__ == "__main__":
    main()