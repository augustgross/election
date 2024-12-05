import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

# Load and preprocess census data
def load_and_process_census_data(census_files):
    combined_data = {}
    
    for year, file_paths in census_files.items():
        for file_path in file_paths:
            # Read the census data
            data = pd.read_csv(file_path, header=None)
            
            # Extract identifier row and select columns ending with 'PE'
            identifiers = data.iloc[0]
            pe_columns = [i for i, id_val in enumerate(identifiers) if str(id_val).endswith('PE')]
            
            # Extract numerical data (3rd row onwards)
            numerical_data = data.iloc[2:, pe_columns]
            numerical_data.columns = identifiers[pe_columns]  # Use identifiers as column names
            
            # Convert to numeric
            numerical_data = numerical_data.apply(pd.to_numeric, errors='coerce')
            
            # Average the numerical data across rows (if multiple rows exist)
            avg_data = numerical_data.mean(axis=0).to_frame().T
            avg_data['Year'] = year
            
            # Append to combined data
            combined_data[year] = avg_data
    
    # Combine all years into a single DataFrame
    all_data = pd.concat(combined_data.values(), ignore_index=True)
    return all_data

# Load presidential election data
def load_pres_data(file_path):
    return pd.read_csv(file_path)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape * 100  # Convert to percentage
    }

# Main pipeline
def main():
    # File paths
    census_data_files = {
        2010: ['census/2010/ACSDP5Y2010.DP05-Data.csv'],
        2012: ['census/2012/ACSDP5Y2012.DP05-Data.csv'],
        2014: ['census/2014/ACSDP5Y2014.DP05-Data.csv'],
        2016: ['census/2016/ACSDP5Y2016.DP05-Data.csv'],
        2018: ['census/2018/ACSDP5Y2018.DP05-Data.csv'],
        2020: ['census/2020/ACSDP5Y2020.DP05-Data.csv'],
        2022: ['census/2022/ACSDP5Y2022.DP05-Data.csv'],
    }
    pres_data_file = 'results/PRES.csv'
    
    # Load and process data
    census_data = load_and_process_census_data(census_data_files)
    pres_data = load_pres_data(pres_data_file)
    
    # Merge census data with presidential election results on 'Year'
    combined_data = pd.merge(pres_data, census_data, on='Year', how='inner')
    
    # Prepare features and target variables
    X = combined_data.drop(columns=['Year', 'Democratic_Pct', 'Republican_Pct', 'Turnout'])
    y_dem = combined_data['Democratic_Pct']
    y_rep = combined_data['Republican_Pct']
    y_turnout = combined_data['Turnout']
    
    # Train-test split
    X_train, X_test, y_dem_train, y_dem_test = train_test_split(X, y_dem, test_size=0.2, random_state=42)
    _, _, y_rep_train, y_rep_test = train_test_split(X, y_rep, test_size=0.2, random_state=42)
    _, _, y_turnout_train, y_turnout_test = train_test_split(X, y_turnout, test_size=0.2, random_state=42)
    
    # Initialize and train XGBoost models
    models = {
        'Democratic_Pct': XGBRegressor(random_state=42),
        'Republican_Pct': XGBRegressor(random_state=42),
        'Turnout': XGBRegressor(random_state=42),
    }
    
    targets = {
        'Democratic_Pct': (y_dem_train, y_dem_test),
        'Republican_Pct': (y_rep_train, y_rep_test),
        'Turnout': (y_turnout_train, y_turnout_test),
    }
    
    for target, model in models.items():
        y_train, y_test = targets[target]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        metrics = calculate_metrics(y_test, predictions)
        
        print(f"\nMetrics for {target}:")
        print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
        print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
        print(f"R-squared (R²): {metrics['R²']:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")

if __name__ == "__main__":
    main()
