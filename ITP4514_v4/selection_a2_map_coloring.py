"""
Map Coloring Algorithm Demo File
Using Constraint Satisfaction Problem (CSP) to Solve Map Coloring Problem
"""

import sys
import os
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from map_coloring_constraint import solve_map_coloring_with_constraint

def main():
    """
    Main function - Run map coloring algorithm demo
    """
    print("ITP4514 - Map Coloring Algorithm Demo")
    print("Author: Li Xueqing")
    print("Date: November 2025")
    print("=" * 50)
    print("Demo Content: Solving Map Coloring Problem Using python-constraint Library")
    print("=" * 50)
    
    try:
        solve_map_coloring_with_constraint()
    except Exception as e:
        print(f"Error running map coloring algorithm demo: {e}")

if __name__ == "__main__":
    main()