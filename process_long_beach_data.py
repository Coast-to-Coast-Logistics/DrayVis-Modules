#!/usr/bin/env python3
"""
Recalculate Long Beach drayage data using US ZIP coordinates
Enhance for Random Forest modeling with bearing degrees and comprehensive features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2 in degrees (0-360)"""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
         np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))
    
    bearing_rad = np.arctan2(y, x)
    bearing_deg = np.degrees(bearing_rad)
    return (bearing_deg + 360) % 360

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in miles"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in miles
    r = 3956
    return c * r

def bearing_to_direction(bearing):
    """Convert bearing to cardinal direction"""
    if bearing >= 337.5 or bearing < 22.5:
        return 'N'
    elif 22.5 <= bearing < 67.5:
        return 'NE'
    elif 67.5 <= bearing < 112.5:
        return 'E'
    elif 112.5 <= bearing < 157.5:
        return 'SE'
    elif 157.5 <= bearing < 202.5:
        return 'S'
    elif 202.5 <= bearing < 247.5:
        return 'SW'
    elif 247.5 <= bearing < 292.5:
        return 'W'
    else:  # 292.5 <= bearing < 337.5
        return 'NW'

def load_and_process_data():
    """Load and process the Long Beach drayage data with ZIP coordinate lookups"""
    
    print("🚛 Processing Long Beach Drayage Data with ZIP Coordinates")
    print("=" * 65)
    
    # Load ZIP coordinates lookup
    print("📍 Loading US ZIP coordinates...")
    zip_coords = pd.read_csv('data/us_zip_coordinates.csv')
    zip_coords['ZIP'] = zip_coords['ZIP'].astype(str).str.zfill(5)  # Ensure 5-digit format
    zip_lookup = dict(zip(zip_coords['ZIP'], zip(zip_coords['LAT'], zip_coords['LNG'])))
    print(f"Loaded {len(zip_lookup):,} ZIP code coordinates")
    
    # Load Long Beach data
    print("\n📊 Loading Long Beach drayage data...")
    df = pd.read_csv('data/8.6.25 LH+F Long Beach with miles.csv')
    print(f"Loaded {len(df)} drayage records")
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Show original columns
    print(f"\nOriginal columns: {list(df.columns)}")
    
    # Clean up date format - remove time
    print("\n🗓️  Cleaning date format...")
    df['Date'] = pd.to_datetime(df['Date Delivery Departure.Date']).dt.date
    
    # Convert ZIP codes to strings with proper formatting
    df['Origin Zip'] = df['Origin Zip'].astype(str).str.zfill(5)
    df['Destination Zip'] = df['Destination Zip'].astype(str).str.zfill(5)
    
    # Lookup coordinates
    print("\n🎯 Looking up ZIP coordinates...")
    def get_coordinates(zip_code):
        if zip_code in zip_lookup:
            return zip_lookup[zip_code]
        return None, None
    
    # Get origin coordinates
    origin_coords = df['Origin Zip'].apply(get_coordinates)
    df['Origin_Lat_New'] = [coord[0] if coord[0] is not None else np.nan for coord in origin_coords]
    df['Origin_Lon_New'] = [coord[1] if coord[1] is not None else np.nan for coord in origin_coords]
    
    # Get destination coordinates
    dest_coords = df['Destination Zip'].apply(get_coordinates)
    df['Dest_Lat_New'] = [coord[0] if coord[0] is not None else np.nan for coord in dest_coords]
    df['Dest_Lon_New'] = [coord[1] if coord[1] is not None else np.nan for coord in dest_coords]
    
    # Check for missing coordinates
    missing_origin = df['Origin_Lat_New'].isna().sum()
    missing_dest = df['Dest_Lat_New'].isna().sum()
    print(f"Missing origin coordinates: {missing_origin}")
    print(f"Missing destination coordinates: {missing_dest}")
    
    # For missing coordinates, use original values if available
    df['Origin_Lat_Final'] = df['Origin_Lat_New'].fillna(df.get('Origin Lat', np.nan))
    df['Origin_Lon_Final'] = df['Origin_Lon_New'].fillna(df.get('Origin Lon', np.nan))
    df['Dest_Lat_Final'] = df['Dest_Lat_New'].fillna(df.get('Dest Lat', np.nan))
    df['Dest_Lon_Final'] = df['Dest_Lon_New'].fillna(df.get('Dest Lon', np.nan))
    
    # Calculate new distances and bearings
    print("\n📏 Recalculating distances and bearings...")
    df['PCMiler_Miles_New'] = calculate_haversine_distance(
        df['Origin_Lat_Final'], df['Origin_Lon_Final'],
        df['Dest_Lat_Final'], df['Dest_Lon_Final']
    )
    
    df['Bearing_New'] = calculate_bearing(
        df['Origin_Lat_Final'], df['Origin_Lon_Final'],
        df['Dest_Lat_Final'], df['Dest_Lon_Final']
    )
    
    # Calculate new RPM
    df['RPM_New'] = df['Linehaul + FSC'] / df['PCMiler_Miles_New']
    
    # Add cardinal direction
    df['Cardinal_Direction'] = df['Bearing_New'].apply(bearing_to_direction)
    
    print("\n✅ Recalculations completed!")
    return df

def enhance_for_random_forest(df):
    """Add enhanced features for Random Forest modeling"""
    
    print("\n🤖 Enhancing data for Random Forest modeling...")
    
    # Geographic features
    df['lat_diff'] = abs(df['Dest_Lat_Final'] - df['Origin_Lat_Final'])
    df['lng_diff'] = abs(df['Dest_Lon_Final'] - df['Origin_Lon_Final'])
    df['route_center_lat'] = (df['Origin_Lat_Final'] + df['Dest_Lat_Final']) / 2
    df['route_center_lng'] = (df['Origin_Lon_Final'] + df['Dest_Lon_Final']) / 2
    
    # Distance-based features
    df['distance_log'] = np.log1p(df['PCMiler_Miles_New'])
    df['distance_squared'] = df['PCMiler_Miles_New'] ** 2
    df['distance_sqrt'] = np.sqrt(df['PCMiler_Miles_New'])
    
    # Rate analysis features
    df['rate_per_mile'] = df['RPM_New']  # Already calculated
    df['rate_per_mile_log'] = np.log1p(df['RPM_New'])
    df['total_revenue'] = df['Linehaul + FSC']
    
    # Time-based features
    df['Date'] = pd.to_datetime(df['Date'])
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Seasonal features
    df['season'] = df['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else
                 'Spring' if x in [3, 4, 5] else
                 'Summer' if x in [6, 7, 8] else 'Fall'
    )
    
    # Distance categories
    df['distance_category'] = pd.cut(df['PCMiler_Miles_New'], 
                                   bins=[0, 25, 50, 100, 200, 500], 
                                   labels=['Short', 'Medium', 'Long', 'Very_Long', 'Ultra_Long'])
    
    # Customer and carrier analysis
    df['customer_encoded'] = pd.Categorical(df['Customer Name']).codes
    df['carrier_encoded'] = pd.Categorical(df['Carrier Name']).codes
    
    # Route frequency (how often this exact route appears)
    route_counts = df.groupby(['Origin Zip', 'Destination Zip']).size()
    df['route_frequency'] = df.apply(lambda row: route_counts[(row['Origin Zip'], row['Destination Zip'])], axis=1)
    
    # Rate efficiency metrics
    df['rate_efficiency'] = df['total_revenue'] / df['PCMiler_Miles_New']
    df['geographic_complexity'] = df['lat_diff'] + df['lng_diff']
    
    # Bearing-based features
    df['bearing_sin'] = np.sin(np.radians(df['Bearing_New']))
    df['bearing_cos'] = np.cos(np.radians(df['Bearing_New']))
    
    print(f"✅ Enhanced features added. Total columns: {len(df.columns)}")
    return df

def create_final_dataset(df):
    """Create the final cleaned and enhanced dataset"""
    
    print("\n📋 Creating final dataset...")
    
    # Select and rename columns for final dataset
    final_columns = {
        'Movement ID': 'movement_id',
        'Date': 'date',
        'Origin City': 'origin_city',
        'Origin State': 'origin_state',
        'Origin Zip': 'origin_zip',
        'Destination City': 'destination_city',
        'Destination State': 'destination_state',
        'Destination Zip': 'destination_zip',
        'Linehaul + FSC': 'total_rate',
        'Carrier Name': 'carrier_name',
        'Carrier ID': 'carrier_id',
        'Customer Name': 'customer_name',
        'Customer ID': 'customer_id',
        'PCMiler_Miles_New': 'miles',
        'Origin_Lat_Final': 'origin_lat',
        'Origin_Lon_Final': 'origin_lng',
        'Dest_Lat_Final': 'destination_lat',
        'Dest_Lon_Final': 'destination_lng',
        'Bearing_New': 'bearing_degree',
        'RPM_New': 'RPM',
        'Cardinal_Direction': 'cardinal_direction'
    }
    
    # Enhanced feature columns
    enhanced_cols = [
        'lat_diff', 'lng_diff', 'route_center_lat', 'route_center_lng',
        'distance_log', 'distance_squared', 'distance_sqrt',
        'rate_per_mile', 'rate_per_mile_log', 'total_revenue',
        'month', 'day_of_week', 'quarter', 'is_weekend', 'season',
        'distance_category', 'customer_encoded', 'carrier_encoded',
        'route_frequency', 'rate_efficiency', 'geographic_complexity',
        'bearing_sin', 'bearing_cos'
    ]
    
    # Create final dataframe
    final_df = df[list(final_columns.keys()) + enhanced_cols].copy()
    final_df = final_df.rename(columns=final_columns)
    
    # Convert date to string format without time
    final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
    
    # Remove rows with missing critical data
    final_df = final_df.dropna(subset=['origin_lat', 'origin_lng', 'destination_lat', 'destination_lng'])
    
    print(f"✅ Final dataset created with {len(final_df)} records and {len(final_df.columns)} columns")
    return final_df

def analyze_improvements(df):
    """Analyze the improvements made to the data"""
    
    print("\n📈 Data Quality Analysis:")
    print("=" * 30)
    
    # Compare original vs new calculations where both exist
    if 'PCMiler Miles' in df.columns:
        original_miles = df['PCMiler Miles'].dropna()
        new_miles = df.loc[original_miles.index, 'PCMiler_Miles_New']
        
        diff = abs(original_miles - new_miles)
        print(f"Distance comparison:")
        print(f"  Average difference: {diff.mean():.2f} miles")
        print(f"  Max difference: {diff.max():.2f} miles")
        print(f"  Correlation: {original_miles.corr(new_miles):.4f}")
    
    # Analyze bearing calculations
    if 'Bearing' in df.columns:
        original_bearing = df['Bearing'].dropna()
        new_bearing = df.loc[original_bearing.index, 'Bearing_New']
        
        # Handle bearing wraparound for comparison
        diff = np.minimum(abs(original_bearing - new_bearing), 
                         360 - abs(original_bearing - new_bearing))
        print(f"\nBearing comparison:")
        print(f"  Average difference: {diff.mean():.2f} degrees")
        print(f"  Max difference: {diff.max():.2f} degrees")
    
    # Data completeness
    print(f"\nData completeness:")
    print(f"  Total records: {len(df):,}")
    print(f"  Complete coordinate records: {df[['Origin_Lat_Final', 'Dest_Lat_Final']].notna().all(axis=1).sum():,}")
    print(f"  Data completeness: {(df[['Origin_Lat_Final', 'Dest_Lat_Final']].notna().all(axis=1).sum() / len(df) * 100):.1f}%")
    
    # Route analysis
    print(f"\nRoute analysis:")
    unique_routes = df.groupby(['Origin Zip', 'Destination Zip']).size()
    print(f"  Unique routes: {len(unique_routes):,}")
    print(f"  Most frequent route: {unique_routes.max()} trips")
    print(f"  Average trips per route: {unique_routes.mean():.1f}")
    
    # Rate analysis
    print(f"\nRate analysis:")
    print(f"  Average rate: ${df['Linehaul + FSC'].mean():.2f}")
    print(f"  Average RPM: ${df['RPM_New'].mean():.2f}")
    print(f"  Rate range: ${df['Linehaul + FSC'].min():.2f} - ${df['Linehaul + FSC'].max():.2f}")

def main():
    """Main processing function"""
    
    # Load and process data
    df = load_and_process_data()
    
    # Analyze improvements
    analyze_improvements(df)
    
    # Enhance for Random Forest
    df_enhanced = enhance_for_random_forest(df)
    
    # Create final dataset
    final_df = create_final_dataset(df_enhanced)
    
    # Save enhanced dataset
    output_file = 'data/long_beach_drayage_enhanced.csv'
    final_df.to_csv(output_file, index=False)
    print(f"\n💾 Enhanced dataset saved to: {output_file}")
    
    # Show sample of final data
    print(f"\n📊 Sample of enhanced data:")
    sample_cols = ['origin_zip', 'destination_zip', 'miles', 'total_rate', 
                   'bearing_degree', 'cardinal_direction', 'RPM', 'season']
    print(final_df[sample_cols].head().to_string(index=False))
    
    # Show feature summary
    print(f"\n🎯 Feature Summary:")
    print(f"Original data columns: 20")
    print(f"Enhanced data columns: {len(final_df.columns)}")
    print(f"New features added: {len(final_df.columns) - 20}")
    
    # Show directional distribution
    print(f"\n🧭 Directional Distribution:")
    direction_counts = final_df['cardinal_direction'].value_counts().sort_index()
    for direction, count in direction_counts.items():
        percentage = (count / len(final_df)) * 100
        print(f"  {direction}: {count:,} routes ({percentage:.1f}%)")
    
    print(f"\n✅ Processing completed successfully!")
    print(f"📁 Enhanced file: {output_file}")
    print(f"🔢 Records: {len(final_df):,}")
    print(f"📊 Features: {len(final_df.columns)}")
    print(f"🚀 Ready for Random Forest modeling!")
    
    return final_df

if __name__ == "__main__":
    enhanced_data = main()
