"""
Route Finding Algorithm Comparison
Running BFS and A* Algorithms and Comparing Results
"""

from route_finding_bfs import RouteFinderBFS
from route_finding_astar import RouteFinderAStar

def compare_algorithms():
    """
    Compare Uninformed Search (BFS) and Informed Search (A*) Algorithm Results
    """
    print("=== Route Finding Algorithm Comparison ===\n")
    
    # Create two finder instances
    bfs_finder = RouteFinderBFS()
    astar_finder = RouteFinderAStar()
    
    # Run BFS algorithm
    print("1. Running Breadth-First Search (BFS)...")
    bfs_path, bfs_distance, bfs_nodes, bfs_time = bfs_finder.bfs_search()
    bfs_finder.print_result(bfs_path, bfs_distance, bfs_nodes, bfs_time)
    
    # Run A* algorithm
    print("2. Running A* Search Algorithm...")
    astar_path, astar_distance, astar_nodes, astar_time = astar_finder.astar_search()
    astar_finder.print_result(astar_path, astar_distance, astar_nodes, astar_time)
    
    # Compare results
    print("=== Algorithm Comparison Analysis ===")
    
    # Path optimality comparison
    if bfs_distance == astar_distance:
        print("✓ Path Optimality: Both algorithms found the shortest path")
    elif bfs_distance < astar_distance:
        print("⚠ Path Optimality: BFS found a shorter path")
    else:
        print("⚠ Path Optimality: A* found a shorter path")
    
    # Time complexity comparison
    print(f"BFS nodes searched: {bfs_nodes}")
    print(f"A* nodes searched: {astar_nodes}")
    
    if bfs_nodes < astar_nodes:
        print("✓ Efficiency: BFS searched fewer nodes")
    elif bfs_nodes > astar_nodes:
        print("✓ Efficiency: A* searched fewer nodes (heuristic function effective)")
    else:
        print("→ Efficiency: Both algorithms searched the same number of nodes")
    
    print(f"BFS execution time: {bfs_time:.6f} seconds")
    print(f"A* execution time: {astar_time:.6f} seconds")
    
    if bfs_time < astar_time:
        print("✓ Execution speed: BFS is faster")
    elif bfs_time > astar_time:
        print("✓ Execution speed: A* is faster")
    else:
        print("→ Execution speed: Both algorithms have the same execution time")

if __name__ == "__main__":
    compare_algorithms()