"""
ITP4514 - AI and Machine Learning Fundamentals
Main Program to Run All Assignment Components
"""

import sys
import os
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from route_finding_comparison import compare_algorithms
from map_coloring_constraint import solve_map_coloring_with_constraint
from sales_prediction import SalesPredictor

def main():
    """
    Main function to run all assignment components
    """
    print("ITP4514 - AI and Machine Learning Fundamentals")
    print("Assignment Demonstration")
    print("Author: Li Xueqing")
    print("Date: November 2025")
    print("=" * 60)
    print("Running all assignment components in sequence...")
    print("=" * 60)
    
    try:
        # Part A1: Route Finding Algorithms
        print("\n" + "=" * 60)
        print("PART A1: ROUTE FINDING ALGORITHMS")
        print("=" * 60)
        compare_algorithms()
        
        # Part A2: Map Coloring CSP
        print("\n" + "=" * 60)
        print("PART A2: MAP COLORING AS CONSTRAINT SATISFACTION PROBLEM")
        print("=" * 60)
        solve_map_coloring_with_constraint()
        
        # Part B: Sales Prediction
        print("\n" + "=" * 60)
        print("PART B: SALES PREDICTION USING LINEAR REGRESSION")
        print("=" * 60)
        predictor = SalesPredictor('sales_data.csv')
        predictor.run_complete_analysis()
        
        print("\n" + "=" * 60)
        print("ALL ASSIGNMENT COMPONENTS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running assignment components: {e}")

if __name__ == "__main__":
    main()