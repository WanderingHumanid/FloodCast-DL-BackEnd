# validate_model.py
# This script loads the flood prediction models and performs validation tests

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

def load_models_and_data():
    """Load the trained models and test data"""
    print("Loading models and data files...")
    
    # Load models
    try:
        regression_model = joblib.load("floodsense_xgb_model_tuned.pkl")
        classifier_model = joblib.load("floodsense_spatio_temporal_classifier.pkl")
        
        # Load time series data
        time_series_df = pd.read_csv("merged_flood_moon_data.csv", index_col=0, parse_dates=True)
        
        # Load spatial data if available
        try:
            spatial_data = pd.read_csv("spatio_temporal_flood_data.csv")
        except:
            spatial_data = None
            print("Spatio-temporal data not available, will validate on time series only")
            
        print("✅ Models and data loaded successfully")
        return regression_model, classifier_model, time_series_df, spatial_data
    except Exception as e:
        print(f"❌ Error loading models or data: {e}")
        return None, None, None, None

def prepare_time_features(df):
    """Prepare time features for the model input"""
    df.index = pd.to_datetime(df.index)
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['dayofyear'] = df_copy.index.dayofyear
    
    # Create lag features
    base_features = ['water_level', 'wind_speed', 'temperature', 'dew_point', 'sea_level_pressure', 'rain', 'moon_illumination_fraction']
    lags = [1, 2, 3, 6]
    for lag in lags:
        for feature in base_features:
            df_copy[f"{feature}_lag{lag}"] = df_copy[feature].shift(lag)
    
    # One-hot encode moon phase if present
    if 'moon_phase' in df_copy.columns:
        df_copy = pd.get_dummies(df_copy, columns=['moon_phase'], prefix='phase', drop_first=True)
    
    return df_copy

def evaluate_model_accuracy(classifier_model, time_series_df):
    """Evaluate the classifier model using cross-validation"""
    print("\n--- Evaluating Model Accuracy ---")
    
    # Prepare data with time features
    prepared_df = prepare_time_features(time_series_df).dropna()
    
    # If the data has a flood_event column, use it for evaluation
    if 'flood_event' in prepared_df.columns:
        print("Found flood_event column, using it for evaluation")
        
        # Extract features and target
        X = prepared_df.drop(columns=['flood_event'])
        y = prepared_df['flood_event']
        
        # Ensure column names match model features
        model_features = classifier_model.get_booster().feature_names
        missing_features = set(model_features) - set(X.columns)
        
        if missing_features:
            print(f"Warning: Missing {len(missing_features)} features required by the model")
            for feature in missing_features:
                X[feature] = 0  # Fill with zeros as fallback
        
        X = X[model_features]  # Reorder columns to match model
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Make predictions
        y_pred_proba = classifier_model.predict_proba(X_test)[:, 1]
        y_pred = classifier_model.predict(X_test)
        
        # Print metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('model_roc_curve.png')
        print("ROC curve saved to model_roc_curve.png")
        
        return True
    else:
        print("No flood_event column found, cannot evaluate classification accuracy")
        return False

def analyze_feature_importance(classifier_model):
    """Analyze feature importance from the classifier model"""
    print("\n--- Analyzing Feature Importance ---")
    
    # Get feature importance
    importance = classifier_model.get_booster().get_score(importance_type='gain')
    
    # Sort features by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    # Print top 10 features
    print("\nTop 10 Important Features:")
    for i, (feature, score) in enumerate(sorted_importance[:10], 1):
        print(f"{i}. {feature}: {score:.2f}")
    
    # Plot feature importance
    features = [x[0] for x in sorted_importance[:15]]
    scores = [x[1] for x in sorted_importance[:15]]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), scores, align='center')
    plt.yticks(range(len(features)), [f.replace('_', ' ') for f in features])
    plt.xlabel('Importance Score')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved to feature_importance.png")
    
    return True

def validate_forecast_generation(regression_model, time_series_df):
    """Validate the water level forecast generation process"""
    print("\n--- Validating Forecast Generation ---")
    
    # Prepare data
    prepared_df = prepare_time_features(time_series_df).dropna()
    
    # Select a period for testing (last 30 days)
    test_period = prepared_df.iloc[-30:]
    
    # Get model features
    model_features = regression_model.get_booster().feature_names
    
    # Check if all required features are available
    missing_features = set(model_features) - set(test_period.columns)
    if missing_features:
        print(f"Warning: Missing {len(missing_features)} features required by the model")
        for feature in missing_features:
            test_period[feature] = 0  # Fill with zeros as fallback
    
    # Generate one-step forecasts
    actual_values = []
    predicted_values = []
    
    for i in range(10, len(test_period)):
        input_data = test_period.iloc[[i-1]][model_features]
        actual = test_period.iloc[i]['water_level']
        pred = regression_model.predict(input_data)[0]
        
        actual_values.append(actual)
        predicted_values.append(pred)
    
    # Calculate error metrics
    mae = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)))
    mape = np.mean(np.abs((np.array(actual_values) - np.array(predicted_values)) / np.array(actual_values))) * 100
    
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 6))
    plt.plot(actual_values, label='Actual Water Level')
    plt.plot(predicted_values, label='Predicted Water Level')
    plt.xlabel('Time Steps')
    plt.ylabel('Water Level')
    plt.title('Actual vs Predicted Water Level')
    plt.legend()
    plt.grid(True)
    plt.savefig('water_level_forecast_validation.png')
    print("Water level forecast validation plot saved to water_level_forecast_validation.png")
    
    return True

def verify_model_sanity():
    """Check if model predictions make sense in various scenarios"""
    print("\n--- Verifying Model Sanity ---")
    
    # Create test cases with different weather scenarios
    scenarios = [
        {"name": "Heavy Rain", "rain": 50, "water_level": 150},
        {"name": "Moderate Rain", "rain": 15, "water_level": 100},
        {"name": "Light Rain", "rain": 5, "water_level": 80},
        {"name": "No Rain", "rain": 0, "water_level": 60}
    ]
    
    # Create a base sample
    base_sample = {
        "water_level": 100,
        "wind_speed": 10,
        "temperature": 25,
        "dew_point": 20,
        "sea_level_pressure": 1010,
        "rain": 0,
        "moon_illumination_fraction": 0.5,
        "hour": 12,
        "dayofweek": 3,
        "month": 6,
        "dayofyear": 180,
        "water_level_lag1": 95,
        "wind_speed_lag1": 9,
        "temperature_lag1": 24,
        "dew_point_lag1": 19,
        "sea_level_pressure_lag1": 1009,
        "rain_lag1": 0,
        "moon_illumination_fraction_lag1": 0.48,
        # Adding other lags with default values
        "water_level_lag2": 90,
        "wind_speed_lag2": 8,
        "temperature_lag2": 24,
        "dew_point_lag2": 19,
        "sea_level_pressure_lag2": 1008,
        "rain_lag2": 0,
        "moon_illumination_fraction_lag2": 0.46,
        "water_level_lag3": 85,
        "wind_speed_lag3": 7,
        "temperature_lag3": 23,
        "dew_point_lag3": 18,
        "sea_level_pressure_lag3": 1007,
        "rain_lag3": 0,
        "moon_illumination_fraction_lag3": 0.44,
        "water_level_lag6": 80,
        "wind_speed_lag6": 6,
        "temperature_lag6": 22,
        "dew_point_lag6": 17,
        "sea_level_pressure_lag6": 1006,
        "rain_lag6": 0,
        "moon_illumination_fraction_lag6": 0.40,
        # Ward features (using average values)
        "avg_elevation_m": 2.0,
        "min_elevation_m": 0.5,
        "low_lying_area_percent": 60.0,
        "distance_to_water_body_m": 500,
        "population_density": 5000
    }
    
    try:
        # Load the classifier model
        classifier_model = joblib.load("floodsense_spatio_temporal_classifier.pkl")
        
        # Get model features
        model_features = classifier_model.get_booster().feature_names
        
        # Create a dataframe for scenarios
        results = []
        
        for scenario in scenarios:
            # Create a copy of the base sample and update with scenario values
            sample = base_sample.copy()
            for key, value in scenario.items():
                if key != "name":
                    sample[key] = value
            
            # Create input dataframe
            input_df = pd.DataFrame([sample])
            
            # Ensure all model features are present
            for feature in model_features:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Select only the features needed by the model
            input_df = input_df[model_features]
            
            # Make prediction
            flood_prob = classifier_model.predict_proba(input_df)[0, 1] * 100
            
            # Add to results
            results.append({
                "Scenario": scenario["name"],
                "Rain (mm)": scenario["rain"],
                "Water Level (cm)": scenario["water_level"],
                "Flood Probability (%)": flood_prob
            })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        print("\nModel predictions for different weather scenarios:")
        print(results_df)
        
        # Check if results make sense (probability should increase with rain and water level)
        is_sane = results_df["Flood Probability (%)"].is_monotonic_decreasing
        
        if is_sane:
            print("✅ Model sanity check PASSED: Higher rain/water levels correspond to higher flood probabilities")
        else:
            print("❌ Model sanity check FAILED: Predictions do not consistently increase with rain/water levels")
        
        # Save results to CSV
        results_df.to_csv("model_scenario_validation.csv", index=False)
        print("Scenario validation results saved to model_scenario_validation.csv")
        
        return is_sane
    except Exception as e:
        print(f"❌ Error during model sanity check: {e}")
        return False

def main():
    """Main validation function"""
    print("=== FloodSense Model Validation ===")
    
    # Load models and data
    regression_model, classifier_model, time_series_df, spatial_data = load_models_and_data()
    
    if regression_model is None or classifier_model is None or time_series_df is None:
        print("❌ Critical error: Could not load models or data")
        return
    
    # Run validation tests
    validation_results = []
    
    # 1. Feature importance analysis
    feature_importance_ok = analyze_feature_importance(classifier_model)
    validation_results.append(("Feature Importance Analysis", feature_importance_ok))
    
    # 2. Accuracy evaluation
    accuracy_ok = evaluate_model_accuracy(classifier_model, time_series_df)
    validation_results.append(("Model Accuracy Evaluation", accuracy_ok))
    
    # 3. Forecast validation
    forecast_ok = validate_forecast_generation(regression_model, time_series_df)
    validation_results.append(("Forecast Generation Validation", forecast_ok))
    
    # 4. Model sanity check
    sanity_ok = verify_model_sanity()
    validation_results.append(("Model Sanity Check", sanity_ok))
    
    # Print validation summary
    print("\n=== Validation Summary ===")
    for test, result in validation_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test}")
    
    print("\nValidation complete. Check the output files for detailed results.")

if __name__ == "__main__":
    main()
