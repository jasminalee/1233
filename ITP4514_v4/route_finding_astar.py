"""
Informed Search Algorithm Implementation - A* Search Algorithm
Used to solve the path finding problem from TSW (Tuen Mun) to TY (Tsuen Wan)
"""

import heapq
import time
import math

class RouteFinderAStar:
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
        
        # Node coordinates (for heuristic function calculation) - These are simplified position estimates
        # In practical applications, real map coordinates can be used
        self.node_coordinates = {
            'TSW': (0, 0),      # Start point
            'YL': (3.89, 0),
            'TM': (3.89 + 7.57, 0),
            'TLC': (3.89 + 7.57, 5.46),
            'LMC': (3.89 + 10.92, 0),
            'LW': (3.89 + 10.92 + 3.89, 0),
            'SS': (3.89 + 10.92 + 3.89 - 1.34, 1.34),
            'KT': (3.89 + 3.89, 0),
            'SK': (3.89 + 3.89 + 8.96, 0),
            'TW': (3.89 + 3.89 + 8.96 + 7.41, 0),
            'TY': (3.89 + 3.89 + 8.96 + 7.41 + 3.43, 0),  # End point
            'KC': (3.89 + 3.89 + 8.96 + 7.41, 3.9)
        }
        
        self.start_node = 'TSW'
        self.goal_node = 'TY'
    
    def heuristic(self, node):
        """
        Heuristic function: Calculate straight-line distance estimate from current node to goal node
        Using Euclidean distance as heuristic function
        """
        if node in self.node_coordinates and self.goal_node in self.node_coordinates:
            x1, y1 = self.node_coordinates[node]
            x2, y2 = self.node_coordinates[self.goal_node]
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return 0.0
    
    def astar_search(self):
        """
        A* Search Algorithm Implementation
        Returns the found path, total distance, and number of nodes searched
        """
        # Record start time
        start_time = time.time()
        
        # Initialize priority queue, storing (f_score, g_score, current node, path)
        # f_score = g_score + heuristic
        # g_score is the actual distance from start node to current node
        frontier = [(float(self.heuristic(self.start_node)), 0.0, self.start_node, [self.start_node])]
        
        # Record visited nodes and their minimum g_score
        explored = {}
        explored[self.start_node] = 0.0
        
        # Record number of nodes searched
        nodes_explored = 0
        
        while frontier:
            # Take the element with minimum f_score from priority queue
            f_score, g_score, current_node, path = heapq.heappop(frontier)
            nodes_explored += 1
            
            # If reached the goal node, return result
            if current_node == self.goal_node:
                end_time = time.time()
                execution_time = end_time - start_time
                return path, g_score, nodes_explored, execution_time
            
            # If we have already found a shorter path to current node, skip
            if g_score > explored.get(current_node, float('inf')):
                continue
            
            # Traverse all neighbors of the current node
            for neighbor, edge_weight in self.graph.get(current_node, []):
                new_g_score = g_score + edge_weight
                # If we found a shorter path to neighbor node
                if new_g_score < explored.get(neighbor, float('inf')):
                    explored[neighbor] = new_g_score
                    new_path = path + [neighbor]
                    f_score = new_g_score + self.heuristic(neighbor)
                    heapq.heappush(frontier, (float(f_score), float(new_g_score), neighbor, new_path))
        
        # If no path is found
        end_time = time.time()
        execution_time = end_time - start_time
        return None, float('inf'), nodes_explored, execution_time
    
    def print_result(self, path, distance, nodes_explored, execution_time):
        """
        Print search results
        """
        print("=== A* Search Algorithm Results ===")
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
    finder = RouteFinderAStar()
    
    # Execute A* search
    path, distance, nodes_explored, execution_time = finder.astar_search()
    
    # Print results
    finder.print_result(path, distance, nodes_explored, execution_time)