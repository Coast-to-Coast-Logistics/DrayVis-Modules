#!/usr/bin/env python3
"""
Quick demonstration of enhanced port drayage data with bearing degrees and total rate analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    else:
        return 'NW'

def enhance_port_drayage_data():
    """Load and enhance the port drayage data with bearing and total rate features"""
    
    print("🚛 Enhancing Port Drayage Data with Bearing Degrees")
    print("=" * 55)
    
    # Load data
    df = pd.read_csv('data/port_drayage_dummy_data.csv')
    print(f"\nLoaded {len(df)} routes from port drayage data")
    
    # Calculate bearing degree
    df['bearing_degree'] = calculate_bearing(
        df['origin_lat'], df['origin_lng'],
        df['destination_lat'], df['destination_lng']
    )
    
    # Add cardinal direction
    df['cardinal_direction'] = df['bearing_degree'].apply(bearing_to_direction)
    
    # Total rate analysis (assuming 'rate' is the total rate)
    df['total_rate'] = df['rate']  # Rename for clarity
    df['rate_per_mile'] = df['total_rate'] / df['miles']
    
    # Geographic features
    df['lat_range'] = abs(df['destination_lat'] - df['origin_lat'])
    df['lng_range'] = abs(df['destination_lng'] - df['origin_lng'])
    
    # Show sample enhanced data
    print("\nSample Enhanced Data:")
    print("-" * 80)
    sample = df[['origin_zip', 'destination_zip', 'miles', 'total_rate', 
                 'bearing_degree', 'cardinal_direction', 'rate_per_mile']].head(10)
    for idx, row in sample.iterrows():
        print(f"Route {idx+1}: {row['origin_zip']} → {row['destination_zip']} | "
              f"{row['miles']:.1f}mi | ${row['total_rate']:.2f} | "
              f"{row['cardinal_direction']} ({row['bearing_degree']:.1f}°) | "
              f"${row['rate_per_mile']:.2f}/mi")
    
    # Analyze bearing patterns
    print(f"\n📊 Bearing Analysis:")
    print("-" * 30)
    
    direction_stats = df.groupby('cardinal_direction').agg({
        'total_rate': ['mean', 'count'],
        'miles': 'mean',
        'rate_per_mile': 'mean'
    }).round(2)
    
    print("Direction | Count | Avg Rate | Avg Miles | Avg $/Mile")
    print("-" * 50)
    for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        if direction in direction_stats.index:
            stats = direction_stats.loc[direction]
            count = int(stats[('total_rate', 'count')])
            avg_rate = stats[('total_rate', 'mean')]
            avg_miles = stats[('miles', 'mean')]
            avg_rpm = stats[('rate_per_mile', 'mean')]
            print(f"{direction:>9} | {count:>5} | ${avg_rate:>7.2f} | {avg_miles:>8.1f} | ${avg_rpm:>7.2f}")
    
    # Rate analysis by distance and direction
    print(f"\n💰 Total Rate Analysis:")
    print("-" * 25)
    print(f"Total Rate Range: ${df['total_rate'].min():.2f} - ${df['total_rate'].max():.2f}")
    print(f"Average Rate: ${df['total_rate'].mean():.2f}")
    print(f"Rate per Mile Range: ${df['rate_per_mile'].min():.2f} - ${df['rate_per_mile'].max():.2f}")
    print(f"Average Rate per Mile: ${df['rate_per_mile'].mean():.2f}")
    
    # Distance-based rate analysis
    df['distance_category'] = pd.cut(df['miles'], 
                                   bins=[0, 25, 50, 100, 200], 
                                   labels=['Short (0-25mi)', 'Medium (25-50mi)', 
                                          'Long (50-100mi)', 'Very Long (100+mi)'])
    
    print(f"\n📏 Rate by Distance Category:")
    print("-" * 35)
    distance_stats = df.groupby('distance_category').agg({
        'total_rate': ['mean', 'count'],
        'rate_per_mile': 'mean'
    }).round(2)
    
    for category in distance_stats.index:
        stats = distance_stats.loc[category]
        count = int(stats[('total_rate', 'count')])
        avg_rate = stats[('total_rate', 'mean')]
        avg_rpm = stats[('rate_per_mile', 'mean')]
        print(f"{category:<20} | {count:>4} routes | ${avg_rate:>6.2f} avg | ${avg_rpm:>5.2f}/mi")
    
    # Create visualizations
    create_enhanced_visualizations(df)
    
    return df

def create_enhanced_visualizations(df):
    """Create visualizations for the enhanced data"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Bearing distribution
    axes[0,0].hist(df['bearing_degree'], bins=36, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Distribution of Route Bearings', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Bearing Degree (°)')
    axes[0,0].set_ylabel('Number of Routes')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Cardinal direction vs total rate
    direction_order = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    sns.boxplot(data=df, x='cardinal_direction', y='total_rate', 
                order=direction_order, ax=axes[0,1])
    axes[0,1].set_title('Total Rate by Cardinal Direction', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Cardinal Direction')
    axes[0,1].set_ylabel('Total Rate ($)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Distance vs total rate with direction colors
    colors = {'N': 'red', 'NE': 'orange', 'E': 'yellow', 'SE': 'green',
              'S': 'blue', 'SW': 'purple', 'W': 'brown', 'NW': 'pink'}
    
    for direction in direction_order:
        direction_data = df[df['cardinal_direction'] == direction]
        axes[1,0].scatter(direction_data['miles'], direction_data['total_rate'], 
                         alpha=0.6, label=direction, color=colors.get(direction, 'gray'))
    
    axes[1,0].set_title('Distance vs Total Rate by Direction', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('Distance (miles)')
    axes[1,0].set_ylabel('Total Rate ($)')
    axes[1,0].legend(title='Direction', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Rate per mile distribution
    axes[1,1].hist(df['rate_per_mile'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,1].set_title('Distribution of Rate per Mile', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('Rate per Mile ($/mile)')
    axes[1,1].set_ylabel('Number of Routes')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('charts/enhanced_bearing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Visualization saved as 'charts/enhanced_bearing_analysis.png'")

def demonstrate_random_forest_features(df):
    """Show how the enhanced features improve Random Forest predictions"""
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    from sklearn.preprocessing import LabelEncoder
    
    print(f"\n🤖 Random Forest Model Enhancement Demo")
    print("=" * 45)
    
    # Prepare features
    features_basic = ['miles', 'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng']
    
    # Enhanced features include bearing
    features_enhanced = features_basic + ['bearing_degree', 'lat_range', 'lng_range', 'rate_per_mile']
    
    # Encode categorical features
    le_carrier = LabelEncoder()
    le_order = LabelEncoder()
    df['carrier_encoded'] = le_carrier.fit_transform(df['carrier'])
    df['order_type_encoded'] = le_order.fit_transform(df['order_type'])
    df['direction_encoded'] = LabelEncoder().fit_transform(df['cardinal_direction'])
    
    features_enhanced += ['carrier_encoded', 'order_type_encoded', 'direction_encoded']
    
    # Target variable
    target = 'total_rate'
    
    # Compare models
    for feature_set, name in [(features_basic, 'Basic'), (features_enhanced, 'Enhanced')]:
        X = df[feature_set].fillna(0)
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"{name} Model ({len(feature_set)} features):")
        print(f"  R² Score: {r2:.3f}")
        print(f"  Mean Absolute Error: ${mae:.2f}")
        
        if name == 'Enhanced':
            # Show feature importance
            importance = pd.DataFrame({
                'feature': feature_set,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"  Top 5 Important Features:")
            for idx, row in importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")
        print()

if __name__ == "__main__":
    # Run the enhancement
    enhanced_df = enhance_port_drayage_data()
    
    # Demonstrate Random Forest improvements
    demonstrate_random_forest_features(enhanced_df)
    
    print("✅ Port drayage data successfully enhanced!")
    print("\nKey enhancements added:")
    print("  🧭 Bearing degree calculations (0-360°)")
    print("  📍 Cardinal direction classification")
    print("  💰 Total rate and rate-per-mile analysis")
    print("  📏 Geographic range features")
    print("  🤖 Improved Random Forest prediction capability")
