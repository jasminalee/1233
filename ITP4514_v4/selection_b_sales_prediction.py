"""
Sales Prediction Algorithm Demo File
Using Simple Linear Regression Algorithm to Predict Future Sales
"""

import sys
import os
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sales_prediction import SalesPredictor

def main():
    """
    Main function - Run sales prediction algorithm demo
    """
    print("ITP4514 - Sales Prediction Algorithm Demo")
    print("Author: Li Xueqing")
    print("Date: November 2025")
    print("=" * 50)
    print("Demo Content: Linear Regression Model for Future Sales Prediction")
    print("=" * 50)
    
    try:
        # Create sales predictor instance and run complete analysis
        predictor = SalesPredictor('sales_data.csv')
        predictor.run_complete_analysis()
        print("\nVisualization charts have been generated:")
        print("1. sales_prediction_results.png - Model evaluation visualization")
        print("2. future_sales_prediction.png - Future sales prediction visualization")
    except Exception as e:
        print(f"Error running sales prediction algorithm demo: {e}")

if __name__ == "__main__":
    main()