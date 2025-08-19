#!/usr/bin/env python3
"""
Random Forest Analysis for Enhanced Long Beach Drayage Data
Comprehensive machine learning analysis with bearing degrees and geographic features
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_enhanced_data():
    """Load the enhanced Long Beach drayage data"""
    
    print("🚛 Loading Enhanced Long Beach Drayage Data")
    print("=" * 50)
    
    df = pd.read_csv('data/long_beach_drayage_enhanced.csv')
    print(f"📊 Loaded {len(df)} records with {len(df.columns)} features")
    
    # Show data overview
    print(f"\n📈 Data Overview:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Rate range: ${df['total_rate'].min():.2f} to ${df['total_rate'].max():.2f}")
    print(f"  Distance range: {df['miles'].min():.1f} to {df['miles'].max():.1f} miles")
    print(f"  RPM range: ${df['RPM'].min():.2f} to ${df['RPM'].max():.2f}")
    
    return df

def prepare_ml_features(df):
    """Prepare features for machine learning"""
    
    print(f"\n🤖 Preparing Machine Learning Features")
    print("=" * 40)
    
    # Features for modeling (exclude target and identifiers)
    exclude_cols = [
        'movement_id', 'date', 'origin_city', 'origin_state', 'origin_zip',
        'destination_city', 'destination_state', 'destination_zip',
        'carrier_name', 'customer_name', 'total_rate',  # target variable
        'carrier_id', 'customer_id', 'season', 'distance_category',
        'cardinal_direction', 'total_revenue'  # categorical/duplicate features
    ]
    
    # Select numerical features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove any remaining non-numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['total_rate']
    
    print(f"✅ Selected {len(X.columns)} numerical features for modeling:")
    print(f"   Target variable: total_rate")
    print(f"   Feature count: {len(X.columns)}")
    
    # Show sample features
    print(f"\n📋 Key Features Selected:")
    key_features = ['miles', 'origin_lat', 'origin_lng', 'destination_lat', 
                   'destination_lng', 'bearing_degree', 'RPM', 'route_frequency']
    for feature in key_features:
        if feature in X.columns:
            print(f"   ✓ {feature}")
    
    return X, y, feature_cols

def train_and_evaluate_models(X, y, feature_names):
    """Train and evaluate different models"""
    
    print(f"\n🎯 Training and Evaluating Models")
    print("=" * 35)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    models = {}
    results = {}
    
    # Linear Regression
    print(f"\n📈 Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    models['Linear Regression'] = lr
    results['Linear Regression'] = {
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R2': r2_score(y_test, lr_pred),
        'MAPE': np.mean(np.abs((y_test - lr_pred) / y_test)) * 100
    }
    
    # Random Forest - Standard
    print(f"🌲 Training Random Forest (Standard)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R2': r2_score(y_test, rf_pred),
        'MAPE': np.mean(np.abs((y_test - rf_pred) / y_test)) * 100
    }
    
    # Random Forest - Enhanced
    print(f"🌳 Training Random Forest (Enhanced)...")
    rf_enhanced = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_enhanced.fit(X_train, y_train)
    rf_enh_pred = rf_enhanced.predict(X_test)
    
    models['Enhanced Random Forest'] = rf_enhanced
    results['Enhanced Random Forest'] = {
        'MAE': mean_absolute_error(y_test, rf_enh_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_enh_pred)),
        'R2': r2_score(y_test, rf_enh_pred),
        'MAPE': np.mean(np.abs((y_test - rf_enh_pred) / y_test)) * 100
    }
    
    return models, results, X_test, y_test

def analyze_feature_importance(models, feature_names):
    """Analyze feature importance from Random Forest models"""
    
    print(f"\n🎯 Feature Importance Analysis")
    print("=" * 30)
    
    # Get feature importance from Enhanced Random Forest
    rf_model = models['Enhanced Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Top 15 features
    print(f"\n🏆 Top 15 Most Important Features:")
    print("-" * 45)
    for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} | {row['importance']:.4f}")
    
    # Geographic/Bearing features
    geo_features = ['bearing_degree', 'cardinal_direction', 'lat_diff', 'lng_diff', 
                   'route_center_lat', 'route_center_lng', 'bearing_sin', 'bearing_cos']
    geo_importance = feature_importance[feature_importance['feature'].isin(geo_features)]
    
    if len(geo_importance) > 0:
        print(f"\n🧭 Geographic/Bearing Feature Importance:")
        print("-" * 40)
        for _, row in geo_importance.iterrows():
            print(f"   {row['feature']:<20} | {row['importance']:.4f}")
    
    return feature_importance

def create_visualizations(results, feature_importance, models, X_test, y_test):
    """Create comprehensive visualizations"""
    
    print(f"\n📊 Creating Visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Long Beach Drayage - Random Forest Analysis', fontsize=16, fontweight='bold')
    
    # 1. Model Performance Comparison
    ax1 = axes[0, 0]
    models_list = list(results.keys())
    r2_scores = [results[model]['R2'] for model in models_list]
    mae_scores = [results[model]['MAE'] for model in models_list]
    
    x_pos = np.arange(len(models_list))
    bars = ax1.bar(x_pos, r2_scores, alpha=0.8, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_xlabel('Models')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Performance (R² Score)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models_list, rotation=45, ha='right')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature Importance (Top 10)
    ax2 = axes[0, 1]
    top_features = feature_importance.head(10)
    y_pos = np.arange(len(top_features))
    bars = ax2.barh(y_pos, top_features['importance'], alpha=0.8, color='#96CEB4')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_features['feature'])
    ax2.set_xlabel('Feature Importance')
    ax2.set_title('Top 10 Feature Importance')
    ax2.invert_yaxis()
    
    # 3. Prediction vs Actual (Enhanced Random Forest)
    ax3 = axes[1, 0]
    rf_enhanced = models['Enhanced Random Forest']
    y_pred = rf_enhanced.predict(X_test)
    
    ax3.scatter(y_test, y_pred, alpha=0.6, color='#FF6B6B', s=30)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, alpha=0.8)
    ax3.set_xlabel('Actual Rate ($)')
    ax3.set_ylabel('Predicted Rate ($)')
    ax3.set_title('Predictions vs Actual (Enhanced RF)')
    
    # Add R² to the plot
    r2_val = results['Enhanced Random Forest']['R2']
    ax3.text(0.05, 0.95, f'R² = {r2_val:.3f}', transform=ax3.transAxes, 
             fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Error Distribution
    ax4 = axes[1, 1]
    errors = y_test - y_pred
    ax4.hist(errors, bins=30, alpha=0.7, color='#45B7D1', edgecolor='black')
    ax4.set_xlabel('Prediction Error ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution (Enhanced RF)')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    
    # Add error statistics
    mae = np.mean(np.abs(errors))
    ax4.text(0.05, 0.95, f'MAE = ${mae:.2f}', transform=ax4.transAxes, 
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('charts/long_beach_random_forest_analysis.png', dpi=300, bbox_inches='tight')
    print(f"💾 Visualization saved: charts/long_beach_random_forest_analysis.png")
    
    return fig

def analyze_route_patterns(df):
    """Analyze route patterns and pricing"""
    
    print(f"\n🛣️  Route Pattern Analysis")
    print("=" * 25)
    
    # Most common routes
    route_freq = df.groupby(['origin_zip', 'destination_zip']).agg({
        'total_rate': ['count', 'mean', 'std'],
        'miles': 'mean',
        'RPM': 'mean',
        'bearing_degree': 'mean'
    }).round(2)
    
    route_freq.columns = ['trip_count', 'avg_rate', 'rate_std', 'avg_miles', 'avg_rpm', 'avg_bearing']
    route_freq = route_freq.sort_values('trip_count', ascending=False)
    
    print(f"🏆 Top 10 Most Frequent Routes:")
    print("-" * 70)
    print(f"{'Route':<18} {'Trips':<6} {'Avg Rate':<10} {'Miles':<8} {'RPM':<8} {'Bearing':<8}")
    print("-" * 70)
    
    for (orig, dest), row in route_freq.head(10).iterrows():
        route_str = f"{orig}→{dest}"
        print(f"{route_str:<18} {row['trip_count']:<6} ${row['avg_rate']:<9.0f} {row['avg_miles']:<7.1f} ${row['avg_rpm']:<7.2f} {row['avg_bearing']:<7.0f}°")
    
    # Directional analysis
    direction_analysis = df.groupby('cardinal_direction').agg({
        'total_rate': ['count', 'mean'],
        'miles': 'mean',
        'RPM': 'mean'
    }).round(2)
    
    direction_analysis.columns = ['trip_count', 'avg_rate', 'avg_miles', 'avg_rpm']
    direction_analysis = direction_analysis.sort_values('trip_count', ascending=False)
    
    print(f"\n🧭 Directional Analysis:")
    print("-" * 50)
    print(f"{'Direction':<10} {'Trips':<6} {'Avg Rate':<10} {'Miles':<8} {'RPM':<8}")
    print("-" * 50)
    for direction, row in direction_analysis.iterrows():
        pct = (row['trip_count'] / len(df)) * 100
        print(f"{direction:<10} {row['trip_count']:<6} ${row['avg_rate']:<9.0f} {row['avg_miles']:<7.1f} ${row['avg_rpm']:<7.2f}")
    
    return route_freq, direction_analysis

def rate_prediction_demo(models, feature_names, df):
    """Demonstrate rate prediction for sample routes"""
    
    print(f"\n💰 Rate Prediction Demo")
    print("=" * 20)
    
    # Get the enhanced model
    model = models['Enhanced Random Forest']
    
    # Sample some routes for prediction
    sample_routes = df.sample(5, random_state=42)
    
    print(f"🎯 Predicting rates for 5 sample routes:")
    print("-" * 60)
    
    for idx, row in sample_routes.iterrows():
        # Prepare features
        feature_values = []
        for feature in feature_names:
            if feature in row:
                feature_values.append(row[feature])
            else:
                feature_values.append(0)  # Default for missing features
        
        # Make prediction
        X_sample = np.array(feature_values).reshape(1, -1)
        predicted_rate = model.predict(X_sample)[0]
        actual_rate = row['total_rate']
        error = abs(predicted_rate - actual_rate)
        error_pct = (error / actual_rate) * 100
        
        print(f"Route: {row['origin_zip']} → {row['destination_zip']}")
        print(f"  Distance: {row['miles']:.1f} miles | Direction: {row['cardinal_direction']} ({row['bearing_degree']:.0f}°)")
        print(f"  Actual: ${actual_rate:.2f} | Predicted: ${predicted_rate:.2f} | Error: ${error:.2f} ({error_pct:.1f}%)")
        print()

def print_performance_summary(results):
    """Print a comprehensive performance summary"""
    
    print(f"\n📊 Model Performance Summary")
    print("=" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n🤖 {model_name}:")
        print(f"   R² Score:    {metrics['R2']:.4f}")
        print(f"   MAE:         ${metrics['MAE']:.2f}")
        print(f"   RMSE:        ${metrics['RMSE']:.2f}")
        print(f"   MAPE:        {metrics['MAPE']:.2f}%")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['R2'])
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   Best R² Score: {results[best_model]['R2']:.4f}")
    print(f"   Best MAE: ${results[best_model]['MAE']:.2f}")

def main():
    """Main analysis function"""
    
    # Load data
    df = load_enhanced_data()
    
    # Prepare features
    X, y, feature_names = prepare_ml_features(df)
    
    # Train models
    models, results, X_test, y_test = train_and_evaluate_models(X, y, feature_names)
    
    # Print performance summary
    print_performance_summary(results)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(models, feature_names)
    
    # Create visualizations
    create_visualizations(results, feature_importance, models, X_test, y_test)
    
    # Route pattern analysis
    route_freq, direction_analysis = analyze_route_patterns(df)
    
    # Rate prediction demo
    rate_prediction_demo(models, feature_names, df)
    
    print(f"\n✅ Random Forest Analysis Complete!")
    print(f"📁 Enhanced dataset: data/long_beach_drayage_enhanced.csv")
    print(f"📊 Analysis chart: charts/long_beach_random_forest_analysis.png")
    print(f"🔢 Records analyzed: {len(df):,}")
    print(f"🎯 Features used: {len(feature_names)}")
    print(f"🚀 Best model accuracy: {max(results.values(), key=lambda x: x['R2'])['R2']:.4f} R²")
    
    return models, results, feature_importance

if __name__ == "__main__":
    models, results, feature_importance = main()
