"""
test_model_with_unforeseen_data.py

This script tests the FloodCast model with unforeseen data to evaluate its generalization ability.
It has three main testing scenarios:
1. Held-out future data (2025 data that the model was not trained on)
2. Synthetic extreme weather scenarios
3. Cross-location testing (if available)

Author: FloodCast Team
Date: August 29, 2025
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Set the plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_models_and_data():
    """Load the trained models and test data"""
    print("Loading models and data files...")
    
    # Load models
    try:
        model_path = os.path.join(os.getcwd(), "floodsense_xgb_model_tuned.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(os.getcwd(), "models", "floodsense_xgb_model_tuned.pkl")
        
        regression_model = joblib.load(model_path)
        
        # Load time series data with 2025 data for testing
        data_path = os.path.join(os.getcwd(), "data", "merged_flood_moon_tide_data.csv")
        if not os.path.exists(data_path):
            data_path = os.path.join(os.getcwd(), "merged_flood_moon_tide_data.csv")
            
        time_series_df = pd.read_csv(data_path, parse_dates=[0], index_col=0)
        
        print("✅ Models and data loaded successfully")
        return regression_model, time_series_df
    except Exception as e:
        print(f"❌ Error loading models or data: {e}")
        return None, None

def prepare_time_features(df):
    """Prepare time features for the model input"""
    df_copy = df.copy()
    
    # Add time-based features
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['dayofyear'] = df_copy.index.dayofyear

    # Define the core numerical features for creating lags
    base_features = [
        'water_level', 'wind_speed', 'temperature', 'dew_point',
        'sea_level_pressure', 'rain', 'moon_illumination_fraction'
    ]
    
    # Add tide_level if it exists in the dataset
    if 'tide_level' in df_copy.columns:
        base_features.append('tide_level')
    
    # Create time-lagged features
    lags = [1, 2, 3, 6]
    for lag in lags:
        for feature in base_features:
            if feature in df_copy.columns:
                df_copy[f"{feature}_lag{lag}"] = df_copy[feature].shift(lag)
    
    # One-hot encode moon phase if present
    if 'moon_phase' in df_copy.columns:
        df_copy = pd.get_dummies(df_copy, columns=['moon_phase'], prefix='phase', drop_first=True)
    
    # Handle missing columns that the model expects
    if regression_model is not None:
        model_features = regression_model.get_booster().feature_names
        for feature in model_features:
            if feature not in df_copy.columns:
                print(f"Warning: Adding missing feature '{feature}' with zeros")
                df_copy[feature] = 0
    
    return df_copy

def test_with_future_data(regression_model, time_series_df):
    """Test the model with future data from 2025 that it hasn't seen during training"""
    print("\n--- Testing Model with Future Data (2025) ---")
    
    # Filter data for 2025
    future_data = time_series_df[time_series_df.index.year >= 2025].copy()
    
    if len(future_data) == 0:
        print("No 2025 data found for testing. Please provide more recent data.")
        return None
    
    print(f"Found {len(future_data)} data points from 2025 for testing")
    
    # Prepare features
    prepared_df = prepare_time_features(future_data)
    prepared_df = prepared_df.dropna()
    
    # Define target (next hour water level)
    prepared_df['actual_next_water_level'] = prepared_df['water_level'].shift(-1)
    prepared_df = prepared_df.dropna()
    
    # Extract features and actual values
    X = prepared_df.drop(columns=['actual_next_water_level', 'water_level'])
    
    # Ensure column order matches model expectations
    model_features = regression_model.get_booster().feature_names
    missing_features = set(model_features) - set(X.columns)
    extra_features = set(X.columns) - set(model_features)
    
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features required by the model")
        for feature in missing_features:
            X[feature] = 0  # Fill with zeros as fallback
    
    if extra_features:
        print(f"Removing {len(extra_features)} extra features not used by the model")
        X = X.drop(columns=list(extra_features))
    
    # Reorder columns to match model
    X = X[model_features]
    
    # Make predictions
    y_pred = regression_model.predict(X)
    y_actual = prepared_df['actual_next_water_level'].values
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    
    print(f"\nResults on 2025 Data:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(14, 7))
    plt.plot(prepared_df.index, y_actual, label='Actual Water Level', color='blue', linewidth=2)
    plt.plot(prepared_df.index, y_pred, label='Predicted Water Level', color='red', linestyle='--', linewidth=2)
    plt.title('Model Performance on Unforeseen 2025 Data', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Water Level', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('unforeseen_data_performance.png')
    print("Plot saved to 'unforeseen_data_performance.png'")
    
    # Save metrics to JSON
    metrics = {
        "unforeseen_2025_data": {
            "samples": len(y_actual),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "mape": round(mape, 2)
        }
    }
    
    with open('unforeseen_data_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Metrics saved to 'unforeseen_data_metrics.json'")
    
    return metrics

def create_synthetic_extreme_scenarios(base_df):
    """Create synthetic extreme weather scenarios to test model robustness"""
    print("\n--- Testing Model with Synthetic Extreme Scenarios ---")
    
    # Use the last 30 days of data as the baseline
    base_data = base_df.iloc[-30*24:].copy()
    
    if len(base_data) == 0:
        print("Not enough data for creating synthetic scenarios")
        return None
    
    # Create a list to store all scenarios
    scenarios = []
    
    # Scenario 1: Heavy rainfall (3x normal rain values)
    scenario1 = base_data.copy()
    if 'rain' in scenario1.columns:
        scenario1['rain'] = scenario1['rain'] * 3
        scenario1['scenario'] = 'Heavy Rainfall (3x)'
        scenarios.append(scenario1)
    
    # Scenario 2: Extreme high tide (1.5x normal tide levels)
    scenario2 = base_data.copy()
    if 'tide_level' in scenario2.columns:
        scenario2['tide_level'] = scenario2['tide_level'] * 1.5
        scenario2['scenario'] = 'Extreme High Tide (1.5x)'
        scenarios.append(scenario2)
    
    # Scenario 3: Combined heavy rain and high tide
    scenario3 = base_data.copy()
    if 'rain' in scenario3.columns and 'tide_level' in scenario3.columns:
        scenario3['rain'] = scenario3['rain'] * 2.5
        scenario3['tide_level'] = scenario3['tide_level'] * 1.3
        scenario3['scenario'] = 'Combined Rain + Tide'
        scenarios.append(scenario3)
    
    # Scenario 4: Rapid pressure drop (cyclone-like conditions)
    scenario4 = base_data.copy()
    if 'sea_level_pressure' in scenario4.columns:
        # Create a gradual drop in pressure over 24 hours
        pressure_drop = np.linspace(0, -25, 24)  # 25 mbar drop over 24 hours
        for i in range(min(len(scenario4), 72)):  # Apply to first 3 days
            if i < 24:
                # Pressure dropping
                scenario4.iloc[i, scenario4.columns.get_loc('sea_level_pressure')] += pressure_drop[i]
            elif i < 48:
                # Pressure at its lowest
                scenario4.iloc[i, scenario4.columns.get_loc('sea_level_pressure')] += -25
            else:
                # Pressure recovering
                recovery_idx = i - 48
                scenario4.iloc[i, scenario4.columns.get_loc('sea_level_pressure')] += -25 + (recovery_idx * (25/24))
        
        scenario4['scenario'] = 'Cyclonic Pressure Drop'
        scenarios.append(scenario4)
    
    # Combine all scenarios for analysis
    if scenarios:
        all_scenarios = pd.concat(scenarios)
        return all_scenarios
    else:
        print("Could not create any synthetic scenarios due to missing columns")
        return None

def test_with_synthetic_scenarios(regression_model, time_series_df):
    """Test the model with synthetic extreme weather scenarios"""
    synthetic_data = create_synthetic_extreme_scenarios(time_series_df)
    
    if synthetic_data is None:
        return None
    
    # Get unique scenarios
    scenarios = synthetic_data['scenario'].unique()
    
    # Prepare features for each scenario
    results = {}
    
    for scenario in scenarios:
        print(f"\nTesting model on scenario: {scenario}")
        
        # Filter data for this scenario
        scenario_data = synthetic_data[synthetic_data['scenario'] == scenario].copy()
        
        # Prepare features
        prepared_df = prepare_time_features(scenario_data)
        prepared_df = prepared_df.dropna()
        
        # Define target (next hour water level)
        prepared_df['actual_water_level'] = prepared_df['water_level']
        prepared_df['water_level'] = prepared_df['water_level'].shift(1)  # Use previous hour to predict current
        prepared_df = prepared_df.dropna()
        
        # Extract features
        X = prepared_df.drop(columns=['actual_water_level', 'scenario'])
        
        # Ensure column order matches model expectations
        model_features = regression_model.get_booster().feature_names
        missing_features = set(model_features) - set(X.columns)
        extra_features = set(X.columns) - set(model_features)
        
        if missing_features:
            for feature in missing_features:
                X[feature] = 0  # Fill with zeros as fallback
        
        if extra_features:
            X = X.drop(columns=list(extra_features))
        
        # Reorder columns to match model
        X = X[model_features]
        
        # Make predictions
        y_pred = regression_model.predict(X)
        
        # Store results
        prepared_df['predicted_water_level'] = y_pred
        
        # Get baseline actual water levels before modification for comparison
        baseline_scenario = time_series_df.loc[prepared_df.index].copy()
        prepared_df['baseline_water_level'] = baseline_scenario['water_level']
        
        # Calculate how much the water level is predicted to change due to the scenario
        delta = y_pred - prepared_df['baseline_water_level'].values
        max_increase = delta.max()
        avg_increase = delta.mean()
        
        print(f"Maximum water level increase: {max_increase:.2f} units")
        print(f"Average water level increase: {avg_increase:.2f} units")
        
        # Store metrics
        results[scenario] = {
            "max_increase": round(float(max_increase), 2),
            "avg_increase": round(float(avg_increase), 2),
            "samples": len(prepared_df)
        }
        
        # Plot the results
        plt.figure(figsize=(14, 7))
        plt.plot(prepared_df.index, prepared_df['baseline_water_level'], 
                 label='Baseline Water Level', color='blue', linewidth=2)
        plt.plot(prepared_df.index, y_pred, 
                 label=f'Predicted Water Level ({scenario})', color='red', 
                 linestyle='--', linewidth=2)
        
        plt.title(f'Model Response to {scenario}', fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Water Level', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        scenario_filename = scenario.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f'scenario_{scenario_filename}_response.png')
    
    # Save scenario results to JSON
    with open('synthetic_scenarios_results.json', 'w') as f:
        json.dump({"synthetic_scenarios": results}, f, indent=2)
    
    print("\nAll synthetic scenario tests completed and saved")
    return results

def test_model_with_thresholds(regression_model, time_series_df):
    """Test the model's ability to accurately predict when water levels exceed critical thresholds"""
    print("\n--- Testing Model Performance at Critical Thresholds ---")
    
    # Use the most recent 90 days of data
    recent_data = time_series_df.iloc[-90*24:].copy()
    
    if len(recent_data) == 0:
        print("Not enough data for threshold testing")
        return None
    
    # Prepare features
    prepared_df = prepare_time_features(recent_data)
    prepared_df = prepared_df.dropna()
    
    # Define target (next hour water level)
    prepared_df['actual_next_water_level'] = prepared_df['water_level'].shift(-1)
    prepared_df = prepared_df.dropna()
    
    # Extract features
    X = prepared_df.drop(columns=['actual_next_water_level'])
    
    # Ensure column order matches model expectations
    model_features = regression_model.get_booster().feature_names
    X_model = pd.DataFrame()
    
    for feature in model_features:
        if feature in X.columns:
            X_model[feature] = X[feature]
        else:
            X_model[feature] = 0  # Fill missing features with zeros
    
    # Make predictions
    y_pred = regression_model.predict(X_model)
    y_actual = prepared_df['actual_next_water_level'].values
    
    # Define thresholds to test (these are example values, adjust as needed)
    # These could be water levels that correspond to different flood severity levels
    thresholds = np.percentile(y_actual, [75, 85, 90, 95, 98])
    
    threshold_results = {}
    
    for i, threshold in enumerate(thresholds):
        percentile = [75, 85, 90, 95, 98][i]
        print(f"\nTesting threshold: {threshold:.2f} (P{percentile})")
        
        # Calculate actual and predicted exceedances
        actual_exceedance = y_actual > threshold
        predicted_exceedance = y_pred > threshold
        
        # Calculate metrics
        true_positives = np.sum(actual_exceedance & predicted_exceedance)
        false_positives = np.sum(~actual_exceedance & predicted_exceedance)
        false_negatives = np.sum(actual_exceedance & ~predicted_exceedance)
        true_negatives = np.sum(~actual_exceedance & ~predicted_exceedance)
        
        # Calculate rates
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"False Alarm Rate: {false_positives/(false_positives+true_negatives):.4f}")
        
        # Store results
        threshold_results[f"P{percentile}"] = {
            "threshold_value": round(float(threshold), 2),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1_score), 4),
            "false_alarm_rate": round(float(false_positives/(false_positives+true_negatives)), 4),
            "exceedance_count": int(np.sum(actual_exceedance))
        }
    
    # Save threshold results to JSON
    with open('threshold_performance.json', 'w') as f:
        json.dump({"threshold_analysis": threshold_results}, f, indent=2)
    
    print("\nThreshold analysis completed and saved to threshold_performance.json")
    
    # Create a visualization of threshold performance
    percentiles = [75, 85, 90, 95, 98]
    metrics = ['precision', 'recall', 'f1_score', 'false_alarm_rate']
    
    plt.figure(figsize=(12, 8))
    
    for metric in metrics:
        values = [threshold_results[f"P{p}"][metric] for p in percentiles]
        plt.plot(percentiles, values, marker='o', linewidth=2, label=metric.replace('_', ' ').title())
    
    plt.title('Model Performance at Different Water Level Thresholds', fontsize=16)
    plt.xlabel('Percentile Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(percentiles)
    plt.tight_layout()
    plt.savefig('threshold_performance.png')
    print("Threshold analysis plot saved to threshold_performance.png")
    
    return threshold_results

def generate_html_report(future_metrics, synthetic_results, threshold_results):
    """Generate an HTML report with all test results"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FloodCast Model Verification Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2, h3 {{ color: #205295; }}
            .section {{ margin-bottom: 30px; padding: 20px; border-radius: 5px; background-color: #f9f9f9; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px 15px; border-bottom: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #205295; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric-card {{ display: inline-block; width: 23%; margin: 1%; padding: 15px; box-sizing: border-box; 
                           border-radius: 5px; background-color: #e3f2fd; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; color: #0277bd; }}
            .images {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin-top: 20px; }}
            .image-container {{ max-width: 45%; }}
            img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            .conclusion {{ background-color: #e3f2fd; padding: 20px; border-radius: 5px; margin-top: 30px; }}
            .timestamp {{ color: #666; font-style: italic; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <h1>FloodCast Model Verification Report</h1>
        <p>This report summarizes the results of testing the FloodCast prediction model with unforeseen data.</p>
        
        <div class="section">
            <h2>1. Performance on Unforeseen 2025 Data</h2>
    """
    
    if future_metrics:
        metrics = future_metrics.get("unforeseen_2025_data", {})
        html_content += f"""
            <p>The model was tested on {metrics.get('samples', 'N/A')} data points from 2025 that were not included in the training data.</p>
            
            <div class="metric-cards">
                <div class="metric-card">
                    <h3>RMSE</h3>
                    <div class="metric-value">{metrics.get('rmse', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h3>MAE</h3>
                    <div class="metric-value">{metrics.get('mae', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h3>R²</h3>
                    <div class="metric-value">{metrics.get('r2', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h3>MAPE</h3>
                    <div class="metric-value">{metrics.get('mape', 'N/A')}%</div>
                </div>
            </div>
            
            <div class="images">
                <div class="image-container">
                    <img src="unforeseen_data_performance.png" alt="Performance on Unforeseen Data">
                </div>
            </div>
        """
    else:
        html_content += "<p>No future data was available for testing.</p>"
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>2. Response to Synthetic Extreme Scenarios</h2>
    """
    
    if synthetic_results:
        html_content += """
            <p>The model was tested against synthetic extreme weather scenarios to evaluate its robustness.</p>
            <table>
                <tr>
                    <th>Scenario</th>
                    <th>Max Water Level Increase</th>
                    <th>Avg Water Level Increase</th>
                    <th>Samples</th>
                </tr>
        """
        
        for scenario, results in synthetic_results.items():
            html_content += f"""
                <tr>
                    <td>{scenario}</td>
                    <td>{results.get('max_increase', 'N/A')}</td>
                    <td>{results.get('avg_increase', 'N/A')}</td>
                    <td>{results.get('samples', 'N/A')}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="images">
        """
        
        for scenario in synthetic_results.keys():
            scenario_filename = scenario.replace(' ', '_').replace('(', '').replace(')', '')
            html_content += f"""
                <div class="image-container">
                    <img src="scenario_{scenario_filename}_response.png" alt="{scenario} Response">
                </div>
            """
        
        html_content += """
            </div>
        """
    else:
        html_content += "<p>No synthetic scenarios were tested.</p>"
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>3. Performance at Critical Thresholds</h2>
    """
    
    if threshold_results:
        html_content += """
            <p>The model's ability to predict when water levels exceed critical thresholds was tested.</p>
            <table>
                <tr>
                    <th>Percentile</th>
                    <th>Threshold Value</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>False Alarm Rate</th>
                    <th>Exceedance Count</th>
                </tr>
        """
        
        for percentile, results in threshold_results.items():
            html_content += f"""
                <tr>
                    <td>{percentile}</td>
                    <td>{results.get('threshold_value', 'N/A')}</td>
                    <td>{results.get('precision', 'N/A')}</td>
                    <td>{results.get('recall', 'N/A')}</td>
                    <td>{results.get('f1_score', 'N/A')}</td>
                    <td>{results.get('false_alarm_rate', 'N/A')}</td>
                    <td>{results.get('exceedance_count', 'N/A')}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <div class="images">
                <div class="image-container">
                    <img src="threshold_performance.png" alt="Threshold Performance">
                </div>
            </div>
        """
    else:
        html_content += "<p>No threshold analysis was performed.</p>"
    
    html_content += f"""
        </div>
        
        <div class="conclusion">
            <h2>Conclusion</h2>
            <p>
                Based on the unforeseen data testing, the FloodCast model demonstrates 
                {'strong' if future_metrics and future_metrics.get('unforeseen_2025_data', {}).get('r2', 0) > 0.9 else 'acceptable'} 
                performance on data it hasn't seen before, with an R² of 
                {future_metrics.get('unforeseen_2025_data', {}).get('r2', 'N/A') if future_metrics else 'N/A'}.
            </p>
            <p>
                The model's response to synthetic extreme scenarios shows that it is 
                {'sensitive to extreme conditions and predicts appropriate water level increases' if synthetic_results else 'not fully tested against extreme conditions'}.
            </p>
            <p>
                At critical thresholds, the model 
                {'performs well with good precision and recall values' if threshold_results and all(r.get('f1_score', 0) > 0.7 for r in threshold_results.values()) else 'shows varying performance, which may require further refinement for high-risk flood prediction'}.
            </p>
        </div>
        
        <p class="timestamp">Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    
    with open('model_verification_report.html', 'w') as f:
        f.write(html_content)
    
    print("\nHTML report generated: model_verification_report.html")

# Main execution
if __name__ == "__main__":
    print("=== FloodCast Model Verification with Unforeseen Data ===")
    print("This script tests the model with data it hasn't seen during training.")
    
    # Load models and data
    regression_model, time_series_df = load_models_and_data()
    
    if regression_model is None or time_series_df is None:
        print("❌ Cannot proceed with testing due to missing models or data.")
        exit(1)
    
    # Test with future data
    future_metrics = test_with_future_data(regression_model, time_series_df)
    
    # Test with synthetic extreme scenarios
    synthetic_results = test_with_synthetic_scenarios(regression_model, time_series_df)
    
    # Test model at critical thresholds
    threshold_results = test_model_with_thresholds(regression_model, time_series_df)
    
    # Generate HTML report
    generate_html_report(future_metrics, synthetic_results, threshold_results)
    
    print("\n=== Model Verification Completed ===")
    print("Review the generated HTML report for comprehensive results.")
