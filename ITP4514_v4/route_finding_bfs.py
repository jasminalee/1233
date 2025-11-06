"""
Uninformed Search Algorithm Implementation - Breadth-First Search (BFS)
Used to solve the path finding problem from TSW (Tuen Mun) to TY (Tsuen Wan)
"""

import collections
import time

class RouteFinderBFS:
    def __init__(self):
        # Nodes and edges in Figure 1 (TSW to TY route network)
        # Nodes: TSW, YL, TM, LMC, KT, TLC, LW, SS, SK, TW, TY, KC
        # Edges and weights (distance unit: kilometers)
        self.graph = {
            'TSW': [('YL', 3.89)],
            'YL': [('TSW', 3.89), ('TM', 7.57), ('LMC', 10.92), ('KT', 3.89)],
            'TM': [('YL', 7.57), ('TLC', 5.46)],
            'TLC': [('TM', 5.46), ('TW', 7.1)],
            'LMC': [('YL', 10.92), ('LW', 3.89), ('SS', 3.42), ('SK', 8.96)],
            'LW': [('LMC', 3.89), ('SS', 1.34)],
            'SS': [('LW', 1.34), ('LMC', 3.42)],
            'KT': [('YL', 3.89), ('SK', 3.43), ('TW', 8.17)],
            'SK': [('KT', 3.43), ('LMC', 8.96), ('TW', 7.41)],
            'TW': [('TLC', 7.1), ('KT', 8.17), ('SK', 7.41), ('KC', 3.9), ('TY', 3.43)],
            'TY': [('TW', 3.43)],
            'KC': [('TW', 3.9)]
        }
        
        self.start_node = 'TSW'
        self.goal_node = 'TY'
    
    def bfs_search(self):
        """
        Breadth-First Search Algorithm Implementation
        Returns the found path, total distance, and number of nodes searched
        """
        # Record start time
        start_time = time.time()
        
        # Initialize queue, storing (current node, path, path total distance)
        queue = collections.deque([(self.start_node, [self.start_node], 0.0)])
        
        # Record visited nodes
        visited = set()
        visited.add(self.start_node)
        
        # Record number of nodes searched
        nodes_explored = 0
        
        while queue:
            # Take the first element from the queue
            current_node, path, distance = queue.popleft()
            nodes_explored += 1
            
            # If reached the goal node, return result
            if current_node == self.goal_node:
                end_time = time.time()
                execution_time = end_time - start_time
                return path, distance, nodes_explored, execution_time
            
            # Traverse all neighbors of the current node
            for neighbor, edge_weight in self.graph.get(current_node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_distance = distance + edge_weight
                    queue.append((neighbor, new_path, new_distance))
        
        # If no path is found
        end_time = time.time()
        execution_time = end_time - start_time
        return None, float('inf'), nodes_explored, execution_time
    
    def print_result(self, path, distance, nodes_explored, execution_time):
        """
        Print search results
        """
        print("=== Breadth-First Search (BFS) Results ===")
        if path:
            print(f"Path found: {' -> '.join(path)}")
            print(f"Total distance: {distance:.2f} kilometers")
            print(f"Nodes searched: {nodes_explored}")
            print(f"Execution time: {execution_time:.6f} seconds")
        else:
            print("No path found from TSW to TY")
        print()

# Main program
if __name__ == "__main__":
    # Create route finder instance
    finder = RouteFinderBFS()
    
    # Execute breadth-first search
    path, distance, nodes_explored, execution_time = finder.bfs_search()
    
    # Print results
    finder.print_result(path, distance, nodes_explored, execution_time)