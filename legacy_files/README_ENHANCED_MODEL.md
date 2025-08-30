# FloodCast Enhanced Model Implementation

## Overview
This implementation integrates the enhanced flood prediction model with tidal data into the FloodCast web application. The enhanced model significantly improves prediction accuracy and lead time:

- **R² Score**: Improved from 0.5899 to 0.9997 (69.5% increase)
- **RMSE**: Reduced from 119.65 cm to 4.26 cm (96.4% reduction)
- **MAE**: Reduced from 80.51 cm to 1.94 cm (97.6% reduction)
- **Lead Time**: Up to 48 hours with R² > 0.87 (previously limited to 4.5 hours)

## Implementation Details

### Configuration
The model selection and configuration settings are stored in `config.py`. To switch between the enhanced model and the original model, set the `USE_ENHANCED_MODEL` flag:

```python
# To use the enhanced model with tidal data
USE_ENHANCED_MODEL = True

# To use the original model without tidal data
USE_ENHANCED_MODEL = False
```

### Key Improvements

1. **Advanced Feature Engineering**:
   - Harmonic time features (sine/cosine transformations)
   - Rolling window statistics (mean, max, min, std) for various time windows
   - Velocity and acceleration features (first and second derivatives)
   - Tide-rain interaction terms
   - Tidal range calculations

2. **Hyperparameter Optimization**:
   - Learning rate: 0.05
   - Max depth: 8
   - Number of estimators: 300
   - Subsample: 0.9

3. **Lead Time Analysis**:
   - Maintains high accuracy (R² > 0.87) for predictions up to 48 hours in advance

## How to Run

1. Make sure the required data files are available:
   - `data/merged_flood_moon_tide_data.csv`: Time series data with tidal information
   - `models/floodsense_xgb_enhanced.pkl`: The enhanced prediction model

2. Start the API server:
   ```
   cd backend
   python app.py
   ```

3. Access the frontend at http://localhost:3000 (assuming your frontend is running on port 3000)

## Data Requirements

The enhanced model requires the following additional features:
- `tide_level`: Tidal height in centimeters
- Derived features: The model will automatically calculate additional features based on the tidal data

## Troubleshooting

If you encounter issues:

1. Check the log file at `logs/app_debug.log`
2. Verify that all required data files exist
3. Ensure that the tidal data CSV contains the necessary columns
4. If necessary, revert to the original model by setting `USE_ENHANCED_MODEL = False` in config.py

For more information about the model improvement process, refer to the outputs of `improve_model.py`.
