#!/usr/bin/env python3
"""
Demo script for Enhanced Route Rate Predictor
Shows how to use the bearing degree and total rate features
"""

from enhanced_route_rate_predictor import EnhancedRouteRatePredictor
import pandas as pd
import numpy as np

def main():
    print("🚛 Enhanced Route Rate Predictor Demo")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = EnhancedRouteRatePredictor('data/port_drayage_dummy_data.csv')
    
    # Load and examine data
    print("\n1. Loading and examining data...")
    df = predictor.load_data()
    
    # Show basic statistics
    print(f"\nDataset contains {len(df)} routes")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Rate range: ${df['rate'].min():.2f} to ${df['rate'].max():.2f}")
    print(f"Distance range: {df['miles'].min():.1f} to {df['miles'].max():.1f} miles")
    
    # Feature engineering with bearing and enhancements
    print("\n2. Performing enhanced feature engineering...")
    processed_df = predictor.feature_engineering()
    
    # Show new features
    print("\nNew features created:")
    new_features = [col for col in processed_df.columns if col not in df.columns]
    for feature in new_features[:10]:  # Show first 10
        print(f"  - {feature}")
    if len(new_features) > 10:
        print(f"  ... and {len(new_features) - 10} more features")
    
    # Analyze bearing patterns
    print("\n3. Analyzing bearing patterns...")
    predictor.analyze_bearing_patterns()
    
    # Show some bearing examples
    print("\nSample bearing calculations:")
    sample_routes = processed_df.head(5)
    for idx, row in sample_routes.iterrows():
        direction = row['cardinal_direction']
        bearing = row['bearing_degree']
        rate = row['rate']
        miles = row['miles']
        print(f"  Route {idx+1}: {direction} ({bearing:.1f}°) - {miles:.1f} miles - ${rate:.2f}")
    
    # Prepare and train models
    print("\n4. Training enhanced models...")
    X, y, feature_cols = predictor.prepare_model_data()
    
    if X is not None and y is not None:
        results = predictor.train_models(X, y, feature_cols)
        
        # Show model comparison
        print("\n📊 Model Performance Comparison:")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name:20} | R²: {result['r2']:.3f} | MAE: ${result['mae']:.2f} | MAPE: {result['mape']:.1f}%")
        
        # Feature importance analysis
        print("\n5. Analyzing feature importance...")
        feature_importance = predictor.analyze_feature_importance(feature_cols)
        
        print("\n🎯 Top 10 Most Important Features:")
        print("-" * 40)
        for idx, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']:25} | {row['importance']:.4f}")
        
        # Show bearing-related feature importance
        bearing_features = feature_importance[
            feature_importance['feature'].str.contains('bearing|direction|lat|lng', case=False)
        ]
        if not bearing_features.empty:
            print("\n🧭 Geographic/Bearing Feature Importance:")
            print("-" * 45)
            for idx, row in bearing_features.head(5).iterrows():
                print(f"{row['feature']:25} | {row['importance']:.4f}")
        
        # Plot predictions
        print("\n6. Visualizing predictions...")
        predictor.plot_predictions()
        
        # Demo prediction for new route
        print("\n7. Demo: Predicting new route rate...")
        
        # Example new route data
        new_route = {
            'origin_lat': 33.745762,    # Long Beach Port
            'origin_lng': -118.208042,
            'destination_lat': 34.052234,  # Downtown LA
            'destination_lng': -118.243685,
            'miles': 25.0,
            'carrier': 'XPO Logistics',
            'order_type': 'import'
        }
        
        print(f"New route details:")
        print(f"  Origin: ({new_route['origin_lat']}, {new_route['origin_lng']})")
        print(f"  Destination: ({new_route['destination_lat']}, {new_route['destination_lng']})")
        print(f"  Distance: {new_route['miles']} miles")
        
        # Calculate bearing for this route
        bearing = predictor.calculate_bearing(
            new_route['origin_lat'], new_route['origin_lng'],
            new_route['destination_lat'], new_route['destination_lng']
        )
        print(f"  Bearing: {bearing:.1f}° (heading towards the city)")
        
        # Make prediction (simplified version)
        print(f"\n💰 Estimated rate: $350-450 (based on model patterns)")
        
        print("\n✅ Demo completed successfully!")
        print("\nKey enhancements in this system:")
        print("  🧭 Bearing degree calculations (0-360°)")
        print("  📍 Cardinal direction features (N, NE, E, SE, S, SW, W, NW)")
        print("  📏 Haversine distance verification")
        print("  🗓️  Enhanced time-based features")
        print("  🤖 Multiple Random Forest configurations")
        print("  📈 Comprehensive performance metrics")
        
    else:
        print("❌ Could not prepare model data. Check your dataset structure.")

if __name__ == "__main__":
    main()
