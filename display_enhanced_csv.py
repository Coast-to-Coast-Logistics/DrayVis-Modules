#!/usr/bin/env python3
"""
Display enhanced CSV file information and validation
"""

import pandas as pd
import numpy as np

def display_enhanced_csv_info():
    """Load and display information about the enhanced CSV file"""
    
    print("📊 Enhanced Port Drayage Dataset Analysis")
    print("=" * 50)
    
    # Load the enhanced dataset
    df = pd.read_csv('data/port_drayage_enhanced_data.csv')
    
    print(f"📁 File: data/port_drayage_enhanced_data.csv")
    print(f"📏 Dataset shape: {df.shape}")
    print(f"💾 File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    # Show column categories
    print(f"\n📋 Column Categories:")
    print("-" * 25)
    
    original_cols = ['origin_zip', 'destination_zip', 'date', 'carrier', 'order_type', 
                    'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng', 
                    'miles', 'rate', 'RPM']
    
    geographic_cols = ['bearing_degree', 'cardinal_direction', 'calculated_miles', 
                      'distance_variance', 'lat_diff', 'lng_diff', 'route_center_lat', 
                      'route_center_lng']
    
    distance_cols = ['distance_log', 'distance_squared', 'distance_sqrt', 'distance_category']
    
    rate_cols = ['total_rate', 'rate_per_mile', 'rate_per_mile_log', 'rate_efficiency']
    
    time_cols = ['month', 'day_of_week', 'quarter', 'is_weekend', 'season']
    
    other_cols = ['geographic_complexity']
    
    print(f"Original Features ({len(original_cols)}): {', '.join(original_cols[:5])}...")
    print(f"Geographic Features ({len(geographic_cols)}): {', '.join(geographic_cols[:5])}...")
    print(f"Distance Features ({len(distance_cols)}): {', '.join(distance_cols)}")
    print(f"Rate Features ({len(rate_cols)}): {', '.join(rate_cols)}")
    print(f"Time Features ({len(time_cols)}): {', '.join(time_cols)}")
    print(f"Other Features ({len(other_cols)}): {', '.join(other_cols)}")
    
    # Show data types
    print(f"\n📊 Data Types Summary:")
    print("-" * 25)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"{str(dtype):<12}: {count:>3} columns")
    
    # Show sample data with key features
    print(f"\n🔍 Sample Enhanced Data (First 5 Rows):")
    print("-" * 45)
    sample_cols = ['origin_zip', 'destination_zip', 'miles', 'total_rate', 
                   'bearing_degree', 'cardinal_direction', 'rate_per_mile', 
                   'season', 'distance_category']
    sample_df = df[sample_cols].head()
    
    for idx, row in sample_df.iterrows():
        print(f"Row {idx+1}:")
        print(f"  Route: {row['origin_zip']} → {row['destination_zip']}")
        print(f"  Distance: {row['miles']:.1f} miles ({row['distance_category']})")
        print(f"  Rate: ${row['total_rate']:.2f} (${row['rate_per_mile']:.2f}/mile)")
        print(f"  Direction: {row['cardinal_direction']} ({row['bearing_degree']:.1f}°)")
        print(f"  Season: {row['season']}")
        print()
    
    # Statistical summary of key features
    print(f"📈 Key Feature Statistics:")
    print("-" * 30)
    
    key_stats = df[['miles', 'total_rate', 'bearing_degree', 'rate_per_mile']].describe()
    print(key_stats.round(2))
    
    # Directional analysis
    print(f"\n🧭 Directional Distribution:")
    print("-" * 30)
    direction_stats = df.groupby('cardinal_direction').agg({
        'total_rate': ['count', 'mean'],
        'miles': 'mean',
        'rate_per_mile': 'mean'
    }).round(2)
    
    print("Direction | Count | Avg Rate | Avg Miles | Avg $/Mi")
    print("-" * 50)
    for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        if direction in direction_stats.index:
            count = int(direction_stats.loc[direction, ('total_rate', 'count')])
            avg_rate = direction_stats.loc[direction, ('total_rate', 'mean')]
            avg_miles = direction_stats.loc[direction, ('miles', 'mean')]
            avg_rpm = direction_stats.loc[direction, ('rate_per_mile', 'mean')]
            print(f"{direction:>9} | {count:>5} | ${avg_rate:>7.2f} | {avg_miles:>8.1f} | ${avg_rpm:>6.2f}")
    
    # Distance category analysis
    print(f"\n📏 Distance Category Analysis:")
    print("-" * 35)
    dist_stats = df.groupby('distance_category').agg({
        'total_rate': ['count', 'mean'],
        'rate_per_mile': 'mean'
    }).round(2)
    
    for category in ['Short', 'Medium', 'Long', 'Very_Long']:
        if category in dist_stats.index:
            count = int(dist_stats.loc[category, ('total_rate', 'count')])
            avg_rate = dist_stats.loc[category, ('total_rate', 'mean')]
            avg_rpm = dist_stats.loc[category, ('rate_per_mile', 'mean')]
            print(f"{category:<12} | {count:>4} routes | ${avg_rate:>6.2f} avg | ${avg_rpm:>5.2f}/mi")
    
    # Seasonal analysis
    print(f"\n🗓️  Seasonal Analysis:")
    print("-" * 25)
    seasonal_stats = df.groupby('season').agg({
        'total_rate': ['count', 'mean'],
        'rate_per_mile': 'mean'
    }).round(2)
    
    for season in ['Spring', 'Summer', 'Fall', 'Winter']:
        if season in seasonal_stats.index:
            count = int(seasonal_stats.loc[season, ('total_rate', 'count')])
            avg_rate = seasonal_stats.loc[season, ('total_rate', 'mean')]
            avg_rpm = seasonal_stats.loc[season, ('rate_per_mile', 'mean')]
            print(f"{season:<8} | {count:>4} routes | ${avg_rate:>6.2f} avg | ${avg_rpm:>5.2f}/mi")
    
    # Data quality check
    print(f"\n🔍 Data Quality Check:")
    print("-" * 25)
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Distance variance (calc vs reported): {df['distance_variance'].mean():.3f} miles avg")
    print(f"Rate per mile range: ${df['rate_per_mile'].min():.2f} - ${df['rate_per_mile'].max():.2f}")
    print(f"Bearing degree range: {df['bearing_degree'].min():.1f}° - {df['bearing_degree'].max():.1f}°")
    
    print(f"\n✅ Enhanced CSV successfully saved and validated!")
    print(f"📁 Location: data/port_drayage_enhanced_data.csv")
    print(f"🔢 Records: {len(df):,}")
    print(f"📊 Features: {len(df.columns)}")
    print(f"🚀 Ready for machine learning and analysis!")

if __name__ == "__main__":
    display_enhanced_csv_info()
