"""
Map Coloring Problem - Implementation Using python-constraint Library
Solving Hong Kong Map Coloring Problem Using python-constraint Package as Required by Assignment
"""

import time
from constraint import Problem, AllDifferentConstraint

def solve_map_coloring_with_constraint():
    """
    Solving Map Coloring Problem Using python-constraint Library
    """
    print("=== Map Coloring Problem Solution (Using python-constraint) ===")
    
    # Areas and adjacency relationships in Figure 2
    # Variables: Areas A, B, C, D, E, F, G, H, I, J, K, L, M, N
    regions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
    
    # Adjacency relationships (adjacent areas cannot have the same color)
    adjacency = {
        'A': ['B', 'N'],
        'B': ['A', 'C', 'D', 'K', 'N'],
        'C': ['B', 'D', 'E', 'G', 'J', 'K'],
        'D': ['B', 'C', 'E'],
        'E': ['C', 'D', 'F', 'G'],
        'F': ['E', 'G', 'H'],
        'G': ['C', 'E', 'F', 'H', 'J'],
        'H': ['F', 'G', 'I', 'J'],
        'I': ['H', 'J'],
        'J': ['C', 'G', 'H', 'I', 'K'],
        'K': ['B', 'C', 'J', 'L'],
        'L': ['K', 'M', 'N'],
        'M': ['L', 'N'],
        'N': ['A', 'B', 'L', 'M']
    }
    
    # Start trying with 2 colors, gradually increase until a solution is found
    for num_colors in range(2, 10):  # Try up to 10 colors
        print(f"Trying to use {num_colors} colors...")
        
        # Record start time
        start_time = time.time()
        
        # Create constraint problem
        problem = Problem()
        
        # Add variables and domains (colors)
        colors = [f"Color{i+1}" for i in range(num_colors)]
        for region in regions:
            problem.addVariable(region, colors)
        
        # Add constraints: Adjacent areas cannot have the same color
        for region, neighbors in adjacency.items():
            for neighbor in neighbors:
                # To avoid adding duplicate constraints (A-B and B-A)
                if region < neighbor:
                    problem.addConstraint(AllDifferentConstraint(), [region, neighbor])
        
        # Solve
        solutions = problem.getSolutions()
        
        # Record end time
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"  Execution time: {execution_time:.6f} seconds")
        
        if solutions:
            print(f"✓ Solution found! Minimum {num_colors} colors required\n")
            
            # Get the first solution
            solution = solutions[0]
            
            # Print solution
            print("=== Map Coloring Solution ===")
            print(f"Minimum number of colors: {num_colors}")
            print(f"Execution time: {execution_time:.6f} seconds")
            print(f"Found {len(solutions)} solutions, displaying the first one:")
            print("\nArea color assignment:")
            
            # Output sorted by area name
            for region in sorted(solution.keys()):
                print(f"  {region}: {solution[region]}")
            
            # Count usage of each color
            color_usage = {}
            for region, color in solution.items():
                if color not in color_usage:
                    color_usage[color] = []
                color_usage[color].append(region)
            
            print("\nColor usage statistics:")
            for color in sorted(color_usage.keys()):
                regions = ', '.join(sorted(color_usage[color]))
                print(f"  {color}: {regions} ({len(color_usage[color])} areas)")
            
            return solution, num_colors, execution_time
    
    print("✗ No solution found")
    return None, 0, 0