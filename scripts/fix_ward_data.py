# fix_ward_data.py
# This script fixes the ward names in the ward_features.csv and kochi_wards.kml files

import pandas as pd
import geopandas as gpd
import os
import numpy as np

def fix_ward_data():
    """Fix the ward name alignment issues in the data files"""
    print("Fixing ward data alignment issues...")
    
    # Load the ward features data
    try:
        ward_features = pd.read_csv('ward_features.csv')
        print(f"Loaded ward features with {len(ward_features)} rows")
        
        # Check for missing ward names
        missing_names = ward_features['ward_name'].isna().sum()
        print(f"Found {missing_names} missing ward names out of {len(ward_features)} rows")
        
        # Load the KML data
        ward_geo = gpd.read_file('kochi_wards.kml', driver='KML')
        print(f"Loaded KML file with {len(ward_geo)} wards")
        
        # Check the names in the KML
        empty_names = ward_geo['Name'].isna().sum() + (ward_geo['Name'] == '').sum()
        print(f"Found {empty_names} empty names in KML file")
        
        # Generate ward names for both datasets
        # We'll use "Ward-1", "Ward-2", etc. for simplicity and clarity
        ward_names = [f"Ward-{i+1}" for i in range(max(len(ward_features), len(ward_geo)))]
        
        # Update the ward features file
        ward_features['ward_name'] = ward_names[:len(ward_features)]
        ward_features.to_csv('ward_features_fixed.csv', index=False)
        print(f"✅ Updated ward_features.csv with {len(ward_features)} named wards")
        
        # Update the KML file with ward names
        # We need to modify the Name property in the GeoDataFrame
        ward_geo['Name'] = ward_names[:len(ward_geo)]
        
        # Save the updated KML file
        ward_geo.to_file('kochi_wards_fixed.kml', driver='KML')
        print(f"✅ Updated kochi_wards.kml with {len(ward_geo)} named wards")
        
        # Create a backup of the original files
        if not os.path.exists('backups'):
            os.makedirs('backups')
        
        # Backup original files
        if os.path.exists('ward_features.csv'):
            os.rename('ward_features.csv', 'backups/ward_features_original.csv')
        
        if os.path.exists('kochi_wards.kml'):
            os.rename('kochi_wards.kml', 'backups/kochi_wards_original.kml')
        
        # Rename the fixed files to the original names
        os.rename('ward_features_fixed.csv', 'ward_features.csv')
        os.rename('kochi_wards_fixed.kml', 'kochi_wards.kml')
        
        print("✅ Updated files have replaced the originals (backups saved in 'backups' folder)")
        return True
    
    except Exception as e:
        print(f"❌ Error fixing ward data: {e}")
        return False

if __name__ == "__main__":
    fix_ward_data()
