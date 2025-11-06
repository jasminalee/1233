"""
Section B Mini Project: Sales Prediction
Using Simple Linear Regression Algorithm to Predict Future Sales
"""

import csv
import math
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class SalesPredictor:
    def __init__(self, data_file):
        """
        Initialize the sales predictor
        """
        self.data_file = data_file
        self.data = []
        
    def load_data(self):
        """
        Load data - data is already weekly aggregated
        """
        print("=== Phase 2: Data Preparation and Analysis ===")
        print("1. Loading data...")
        
        # Read all data
        raw_data = []
        with open(self.data_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert data types
                row['sales'] = float(row['sales']) if row['sales'] != 'NA' else 0
                row['temperature'] = float(row['temperature']) if row['temperature'] != 'NA' else 0
                row['is_holiday'] = int(row['is_holiday']) if row['is_holiday'] != 'NA' else 0
                row['promotion'] = int(row['promotion']) if row['promotion'] != 'NA' else 0
                row['date'] = datetime.strptime(row['date'], '%Y/%m/%d')
                raw_data.append(row)
        
        # Since data is already weekly, we just need to remove duplicates and keep one record per week
        print("2. Processing weekly data...")
        date_groups = defaultdict(list)
        for row in raw_data:
            date_groups[row['date']].append(row)
        
        # For each date, we'll take the first record (assuming they're the same for the same date)
        for date, rows in date_groups.items():
            # Take the first row for each date
            row = rows[0]
            self.data.append({
                'date': date,
                'sales': row['sales'],
                'temperature': row['temperature'],
                'is_holiday': row['is_holiday'],
                'promotion': row['promotion']
            })
        
        # Sort by date
        self.data.sort(key=lambda x: x['date'])
        
        print(f"Successfully loaded {len(self.data)} weekly records from {len(raw_data)} raw records")
        print(f"Data columns: {list(self.data[0].keys())}")
        
        # Display first few data points
        print("\nFirst 5 weekly data points:")
        for i in range(min(5, len(self.data))):
            print(f"  Date: {self.data[i]['date'].strftime('%Y-%m-%d')}, Sales: {self.data[i]['sales']:,.2f}, "
                  f"Temp: {self.data[i]['temperature']:.2f}, Holiday: {self.data[i]['is_holiday']}, "
                  f"Promo: {self.data[i]['promotion']}")
        
    def exploratory_analysis(self):
        """
        Simple data analysis
        """
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic statistics
        sales = [row['sales'] for row in self.data]
        temperatures = [row['temperature'] for row in self.data]
        holidays = [row['is_holiday'] for row in self.data]
        promotions = [row['promotion'] for row in self.data]
        
        print(f"Weekly Sales Statistics:")
        print(f"  Minimum: {min(sales):,.2f}")
        print(f"  Maximum: {max(sales):,.2f}")
        print(f"  Average: {sum(sales) / len(sales):,.2f}")
        
        print(f"\nTemperature Statistics:")
        print(f"  Minimum: {min(temperatures):.2f}")
        print(f"  Maximum: {max(temperatures):.2f}")
        print(f"  Average: {sum(temperatures) / len(temperatures):.2f}")
        
        print(f"\nHoliday Distribution:")
        holiday_count = sum(holidays)
        print(f"  Weeks with holidays: {holiday_count}")
        print(f"  Weeks without holidays: {len(holidays) - holiday_count}")
        
        print(f"\nPromotion Distribution:")
        promo_count = sum(promotions)
        print(f"  Weeks with promotions: {promo_count}")
        print(f"  Weeks without promotions: {len(promotions) - promo_count}")
        
        # Simple correlation analysis
        print(f"\nSimple Correlation Analysis:")
        # Relationship between holidays and sales
        holiday_sales = [row['sales'] for row in self.data if row['is_holiday'] == 1]
        non_holiday_sales = [row['sales'] for row in self.data if row['is_holiday'] == 0]
        
        avg_holiday_sales = sum(holiday_sales) / len(holiday_sales) if holiday_sales else 0
        avg_non_holiday_sales = sum(non_holiday_sales) / len(non_holiday_sales) if non_holiday_sales else 0
        
        print(f"  Average sales in weeks with holidays: {avg_holiday_sales:,.2f}")
        print(f"  Average sales in weeks without holidays: {avg_non_holiday_sales:,.2f}")
        
        # Relationship between promotions and sales
        promo_sales = [row['sales'] for row in self.data if row['promotion'] == 1]
        non_promo_sales = [row['sales'] for row in self.data if row['promotion'] == 0]
        
        avg_promo_sales = sum(promo_sales) / len(promo_sales) if promo_sales else 0
        avg_non_promo_sales = sum(non_promo_sales) / len(non_promo_sales) if non_promo_sales else 0
        
        print(f"  Average sales in weeks with promotions: {avg_promo_sales:,.2f}")
        print(f"  Average sales in weeks without promotions: {avg_non_promo_sales:,.2f}")
        
    def prepare_features(self):
        """
        Prepare features
        """
        print("\n=== Feature Preparation ===")
        
        # Extract features
        self.features = []
        self.targets = []
        
        for row in self.data:
            # Features: Temperature, Is Holiday, Promotion
            feature = [
                row['temperature'],
                row['is_holiday'],
                row['promotion']
            ]
            self.features.append(feature)
            self.targets.append(row['sales'])
        
        print(f"Feature matrix size: {len(self.features)} x {len(self.features[0])}")
        print(f"Target variable size: {len(self.targets)}")
        print("Feature meanings:")
        print("  1. Average Temperature")
        print("  2. Is Holiday (0=No, 1=Yes)")
        print("  3. Promotion (0=No, 1=Yes)")
        
    def linear_regression(self):
        """
        Simple linear regression implementation
        """
        print("\n=== Phase 3: Solution Design ===")
        print("Selecting linear regression model for sales prediction")
        print("Model formula: y = w0 + w1*x1 + w2*x2 + w3*x3")
        print("Where:")
        print("  y = Weekly Sales")
        print("  x1 = Average Temperature")
        print("  x2 = Is Holiday")
        print("  x3 = Promotion")
        
        print("\n=== Phase 4: Solution Implementation ===")
        print("1. Training linear regression model using least squares method...")
        
        # Add bias term (x0 = 1)
        X = [[1] + feature for feature in self.features]
        y = self.targets
        
        # Simple linear regression implementation
        # Using normal equation: w = (X^T * X)^(-1) * X^T * y
        n = len(X)
        m = len(X[0])
        
        # Calculate X^T * X
        XT_X = [[0 for _ in range(m)] for _ in range(m)]
        for i in range(m):
            for j in range(m):
                for k in range(n):
                    XT_X[i][j] += X[k][i] * X[k][j]
        
        # Calculate X^T * y
        XT_y = [0 for _ in range(m)]
        for i in range(m):
            for k in range(n):
                XT_y[i] += X[k][i] * y[k]
        
        # Solve linear system (simplified implementation)
        self.weights = self.solve_linear_system(XT_X, XT_y)
        
        print("2. Model training completed")
        print(f"Model parameters: {self.weights}")
        print("Parameter meanings:")
        print(f"  w0 (Bias): {self.weights[0]:,.2f}")
        print(f"  w1 (Temperature coefficient): {self.weights[1]:.2f}")
        print(f"  w2 (Holiday coefficient): {self.weights[2]:.2f}")
        print(f"  w3 (Promotion coefficient): {self.weights[3]:.2f}")
        
    def solve_linear_system(self, A, b):
        """
        Simplified linear system solver
        Using Gaussian elimination
        """
        n = len(A)
        
        # Create augmented matrix
        augmented = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
        
        # Forward elimination
        for i in range(n):
            # Select pivot
            max_row = i
            for k in range(i+1, n):
                if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                    max_row = k
            
            # Swap rows
            augmented[i], augmented[max_row] = augmented[max_row], augmented[i]
            
            # Elimination
            for k in range(i+1, n):
                if augmented[i][i] != 0:
                    factor = augmented[k][i] / augmented[i][i]
                    for j in range(i, n+1):
                        augmented[k][j] -= factor * augmented[i][j]
        
        # Back substitution
        x = [0 for _ in range(n)]
        for i in range(n-1, -1, -1):
            x[i] = augmented[i][n]
            for j in range(i+1, n):
                x[i] -= augmented[i][j] * x[j]
            if augmented[i][i] != 0:
                x[i] /= augmented[i][i]
        
        return x
    
    def evaluate_model(self):
        """
        Evaluate model
        """
        print("\n=== Model Evaluation ===")
        
        # Predict all samples
        predictions = []
        X = [[1] + feature for feature in self.features]
        
        for i in range(len(X)):
            pred = sum(X[i][j] * self.weights[j] for j in range(len(self.weights)))
            predictions.append(pred)
        
        # Calculate evaluation metrics
        y = self.targets
        
        # MAE (Mean Absolute Error)
        mae = sum(abs(y[i] - predictions[i]) for i in range(len(y))) / len(y)
        
        # RMSE (Root Mean Square Error)
        rmse = math.sqrt(sum((y[i] - predictions[i])**2 for i in range(len(y))) / len(y))
        
        # R² (Coefficient of Determination)
        y_mean = sum(y) / len(y)
        ss_tot = sum((y[i] - y_mean)**2 for i in range(len(y)))
        ss_res = sum((y[i] - predictions[i])**2 for i in range(len(y)))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print(f"Model Performance Metrics:")
        print(f"  MAE (Mean Absolute Error): {mae:,.2f}")
        print(f"  RMSE (Root Mean Square Error): {rmse:,.2f}")
        print(f"  R² (Coefficient of Determination): {r2:.4f}")
        
        # Display some prediction results
        print(f"\nFirst 10 sample prediction results:")
        print("Actual\t\tPredicted\tError")
        for i in range(min(10, len(y))):
            error = abs(y[i] - predictions[i])
            print(f"{y[i]:,.2f}\t\t{predictions[i]:,.2f}\t{error:,.2f}")
            
        # Save prediction results for visualization
        self.predictions = predictions
        
    def predict_future_sales(self):
        """
        Predict future sales
        """
        print("\n=== Future Sales Prediction ===")
        
        # Create sample data for the next few weeks
        last_date = self.data[-1]['date']
        future_data = []
        
        # Generate data for next 5 weeks
        for i in range(1, 6):
            future_date = last_date + timedelta(weeks=i)
            # Use average values from existing data for prediction
            import random
            avg_temp = random.uniform(40, 75)            # Alternate holiday and promotion for variety
            is_holiday = 1 if i % 3 == 0 else 0
            is_promotion = 1 if i % 2 == 0 else 0
            
            future_data.append([avg_temp, is_holiday, is_promotion])
        
        print("Future 5 weeks prediction data:")
        print("Avg Temp\tHoliday\tPromo")
        for data in future_data:
            print(f"{data[0]:.2f}\t\t{data[1]}\t{data[2]}")
        
        # Use model for prediction
        print(f"\nPrediction results using linear regression model:")
        self.future_predictions = []
        for i, data in enumerate(future_data):
            # Add bias term
            x = [1] + data
            # Calculate prediction
            prediction = sum(x[j] * self.weights[j] for j in range(len(self.weights)))
            self.future_predictions.append(prediction)
            future_date = last_date + timedelta(weeks=i+1)
            print(f"  Week of {future_date.strftime('%Y-%m-%d')} predicted sales: {prediction:,.2f}")
        
    def visualize_results(self):
        """
        Visualize prediction results
        """
        print("\n=== Visualization Results ===")
        
        # Use system default fonts to avoid Chinese font warnings
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure and subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # 1. Actual vs Predicted scatter plot
        ax1.scatter(self.targets, self.predictions, alpha=0.7)
        ax1.plot([min(self.targets), max(self.targets)], [min(self.targets), max(self.targets)], 'r--', lw=2)
        ax1.set_xlabel('Actual Weekly Sales')
        ax1.set_ylabel('Predicted Weekly Sales')
        ax1.set_title('Actual vs Predicted Weekly Sales')
        ax1.grid(True, alpha=0.3)
        
        # Format axis labels with commas for thousands
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Add R² value to the plot
        y_mean = sum(self.targets) / len(self.targets)
        ss_tot = sum((y - y_mean) ** 2 for y in self.targets)
        ss_res = sum((y - pred) ** 2 for y, pred in zip(self.targets, self.predictions))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        ax1.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax1.transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Time series plot: Actual vs Predicted sales
        dates = [row['date'] for row in self.data]
        ax2.plot(dates, self.targets, label='Actual Weekly Sales', marker='o', linewidth=2)
        ax2.plot(dates, self.predictions, label='Predicted Weekly Sales', marker='s', linewidth=2)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Weekly Sales')
        ax2.set_title('Weekly Sales Time Series Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis with commas for thousands
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Feature importance bar chart
        feature_names = ['Temperature', 'Holiday', 'Promotion']
        feature_weights = [abs(self.weights[i+1]) for i in range(3)]  # Exclude bias term
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        bars = ax3.bar(feature_names, feature_weights, color=colors)
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Importance (|Weight|)')
        ax3.set_title('Feature Importance')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, weight in zip(bars, feature_weights):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{weight:,.0f}', ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig('sales_prediction_results.png', dpi=300, bbox_inches='tight')
        print("Results saved as 'sales_prediction_results.png'")
        
        # Close figure to release memory
        plt.close()
        
    def visualize_future_predictions(self):
        """
        Visualize future sales predictions
        """
        if not hasattr(self, 'future_predictions'):
            print("Please run predict_future_sales() method first")
            return
            
        # Use system default fonts to avoid Chinese font warnings
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Get last few weeks of actual data for comparison
        last_dates = [row['date'] for row in self.data[-10:]]
        last_sales = [row['sales'] for row in self.data[-10:]]
        
        # Create future dates
        last_date = self.data[-1]['date']
        future_dates = [last_date + timedelta(weeks=i+1) for i in range(len(self.future_predictions))]
        
        # Plot chart
        plt.plot(last_dates, last_sales, marker='o', label='Historical Weekly Sales', linewidth=2)
        plt.plot(future_dates, self.future_predictions, marker='s', label='Predicted Weekly Sales', linewidth=2, linestyle='--')
        
        # Add data point labels
        for i, (date, pred) in enumerate(zip(future_dates, self.future_predictions)):
            plt.annotate(f'{pred:,.0f}', (date, pred), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.xlabel('Date')
        plt.ylabel('Weekly Sales')
        plt.title('Future Weekly Sales Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format y-axis with commas for thousands
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('future_sales_prediction.png', dpi=300, bbox_inches='tight')
        print("Future prediction results saved as 'future_sales_prediction.png'")
        
        # Close figure to release memory
        plt.close()
    
    def run_complete_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("ITP4514 - Section B Mini Project: Sales Prediction")
        print("=" * 50)
        
        # Phase 1: Problem Analysis
        print("\n=== Phase 1: Problem Analysis ===")
        print("Business Problem: Predict future weekly sales based on historical sales data, weather information, holidays, and promotional activities")
        print("Project Goal: Build a machine learning model to accurately predict future weekly sales and support business decision-making")
        print("Expected Outcomes: Implement an accurate sales prediction system to help optimize inventory management and marketing strategies")
        
        # Execute all phases
        self.load_data()
        self.exploratory_analysis()
        self.prepare_features()
        self.linear_regression()
        self.evaluate_model()
        self.predict_future_sales()
        
        # Visualize results
        self.visualize_results()
        self.visualize_future_predictions()
        
        print("\n" + "=" * 50)
        print("Sales Prediction Project Completed")
        print("=" * 50)

# Main execution
if __name__ == "__main__":
    predictor = SalesPredictor("sales_data.csv")
    predictor.run_complete_analysis()