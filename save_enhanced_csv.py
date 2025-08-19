#!/usr/bin/env python3
"""
Save enhanced port drayage data with bearing degrees and additional features to new CSV
"""

import pandas as pd
import numpy as np

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

def enhance_and_save_data():
    """Load, enhance, and save the port drayage data"""
    
    print("📂 Loading original port drayage data...")
    df = pd.read_csv('data/port_drayage_dummy_data.csv')
    print(f"Loaded {len(df)} routes with {len(df.columns)} columns")
    
    print("\n🔧 Adding enhanced features...")
    
    # Calculate bearing degree
    df['bearing_degree'] = calculate_bearing(
        df['origin_lat'], df['origin_lng'],
        df['destination_lat'], df['destination_lng']
    )
    
    # Add cardinal direction
    df['cardinal_direction'] = df['bearing_degree'].apply(bearing_to_direction)
    
    # Calculate verified distance using Haversine formula
    df['calculated_miles'] = calculate_haversine_distance(
        df['origin_lat'], df['origin_lng'],
        df['destination_lat'], df['destination_lng']
    )
    
    # Calculate distance variance (difference between reported and calculated)
    df['distance_variance'] = abs(df['miles'] - df['calculated_miles'])
    
    # Geographic features
    df['lat_diff'] = abs(df['destination_lat'] - df['origin_lat'])
    df['lng_diff'] = abs(df['destination_lng'] - df['origin_lng'])
    
    # Route center coordinates (midpoint)
    df['route_center_lat'] = (df['origin_lat'] + df['destination_lat']) / 2
    df['route_center_lng'] = (df['origin_lng'] + df['destination_lng']) / 2
    
    # Distance-based features
    df['distance_log'] = np.log1p(df['miles'])
    df['distance_squared'] = df['miles'] ** 2
    df['distance_sqrt'] = np.sqrt(df['miles'])
    
    # Rate analysis features
    df['total_rate'] = df['rate']  # Rename for clarity
    df['rate_per_mile'] = df['total_rate'] / df['miles']
    df['rate_per_mile_log'] = np.log1p(df['rate_per_mile'])
    
    # Time-based features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Seasonal features
    df['season'] = df['month'].apply(
        lambda x: 'Winter' if x in [12, 1, 2] else
                 'Spring' if x in [3, 4, 5] else
                 'Summer' if x in [6, 7, 8] else 'Fall'
    )
    
    # Distance categories
    df['distance_category'] = pd.cut(df['miles'], 
                                   bins=[0, 25, 50, 100, 200], 
                                   labels=['Short', 'Medium', 'Long', 'Very_Long'])
    
    # Rate efficiency metrics
    df['rate_efficiency'] = df['total_rate'] / df['calculated_miles']
    df['geographic_complexity'] = df['lat_diff'] + df['lng_diff']
    
    print(f"✅ Enhanced data now has {len(df.columns)} columns")
    
    # Show sample of new features
    print("\n📊 Sample of enhanced data:")
    sample_cols = ['origin_zip', 'destination_zip', 'miles', 'total_rate', 
                   'bearing_degree', 'cardinal_direction', 'rate_per_mile', 'season']
    print(df[sample_cols].head(3).to_string(index=False))
    
    # Save the enhanced dataset
    output_file = 'data/port_drayage_enhanced_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Enhanced data saved to: {output_file}")
    
    # Show summary statistics
    print(f"\n📈 Enhancement Summary:")
    print(f"Original columns: {len(pd.read_csv('data/port_drayage_dummy_data.csv').columns)}")
    print(f"Enhanced columns: {len(df.columns)}")
    print(f"New features added: {len(df.columns) - len(pd.read_csv('data/port_drayage_dummy_data.csv').columns)}")
    
    print(f"\nNew features include:")
    new_features = [col for col in df.columns if col not in pd.read_csv('data/port_drayage_dummy_data.csv').columns]
    for i, feature in enumerate(new_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # Show directional distribution
    print(f"\n🧭 Directional Distribution:")
    direction_counts = df['cardinal_direction'].value_counts().sort_index()
    for direction, count in direction_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {direction}: {count:,} routes ({percentage:.1f}%)")
    
    return df, output_file

if __name__ == "__main__":
    enhanced_df, saved_file = enhance_and_save_data()
    print(f"\n✅ Successfully created enhanced dataset: {saved_file}")
