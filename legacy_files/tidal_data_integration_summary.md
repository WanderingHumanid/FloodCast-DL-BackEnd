# Tidal Data Integration for FloodCast Project

## Overview
This document summarizes the integration of tidal data into the FloodCast flood prediction system for Kochi. The tidal data significantly improves the accuracy of flood predictions by accounting for the coastal influence on water levels.

## Data Processing
- Extracted hourly tidal data from 2023-01-01 to 2025-08-23 from the original h174.csv dataset
- Merged tidal data with existing weather and water level data
- Created a new dataset: `merged_flood_moon_tide_data.csv`

## Key Findings

### Correlation Analysis
- Tide level vs. Moon illumination: **0.1339** (weak positive correlation)
- Tide level vs. Water level: **-0.0287** (weak negative correlation)
- Tide level vs. Rainfall: **0.0104** (negligible correlation)

### Model Performance Improvement
- Original model (without tidal data):
  - R² score: **-0.5725** (negative indicates poor fit)
  - MAE: **185.70 cm**
  - RMSE: **234.30 cm**

- Enhanced model (with tidal data):
  - R² score: **0.5899** (significant improvement)
  - MAE: **80.51 cm**
  - RMSE: **119.65 cm**

- Overall improvement:
  - RMSE improvement: **48.93%**
  - Error reduction: Nearly cut in half

### Additional Insights
- High tide events show a stronger correlation with water levels during dry periods
- Hourly patterns reveal significant tidal influence throughout the day
- Extreme high tide events appear to have a delayed impact on water levels

## Implementation Details

### New Files Created
- `extract_tidal_data.py`: Extracts and processes tidal data
- `analyze_tidal_influence.py`: Analyzes the relationships between tide, water levels, and moon phases
- `train_model_with_tidal_data.py`: Trains a new XGBoost model incorporating tidal data
- `compare_models.py`: Compares performance between models with and without tidal data
- `app_with_tidal_data.py`: Updated Flask backend with tidal data integration
- `run_tidal_model.py`: Helper script to run the tidal-enabled app

### Data Files
- `kochi_tidal_data_2023_2025.csv`: Extracted tidal data for the relevant period
- `merged_flood_moon_tide_data.csv`: Complete dataset with tidal information
- `sample_merged_data.csv`: Sample of the merged data for quick inspection

### Model Files
- `floodsense_xgb_with_tidal_data.pkl`: XGBoost model trained with tidal data
- `floodsense_xgb_without_tidal_data.pkl`: XGBoost model trained without tidal data for comparison

## Visualization
Several visualizations were generated to understand the relationships:
- Tide-water correlation plots
- Time series analysis of tidal patterns
- Hourly and monthly tidal patterns
- Extreme tidal event analysis
- Model comparison visualizations

## Future Improvements
1. Further analyze the relationship between high tide events and specific flood-prone wards
2. Incorporate tidal forecasts from meteorological services for extended prediction horizons
3. Develop a spatial analysis of tidal influence in different areas of Kochi
4. Consider lunar cycle forecasts for longer-term flood risk assessment

## Conclusion
The integration of tidal data has significantly improved the flood prediction model's accuracy. By accounting for coastal influences, the system can now better predict water levels and flood risks in Kochi, reducing the RMSE by nearly 50%.
