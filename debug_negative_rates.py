#!/usr/bin/env python3
"""
Debug the negative rate prediction issue
"""

import pandas as pd
import numpy as np
from advanced_rate_estimator import AdvancedLAPortRateEstimator

def debug_rate_prediction():
    print("🔍 DEBUGGING NEGATIVE RATE PREDICTION ISSUE")
    print("=" * 60)
    
    try:
        # Initialize estimator
        estimator = AdvancedLAPortRateEstimator()
        
        # Check training data
        print("\n📊 TRAINING DATA ANALYSIS:")
        print(f"Data shape: {estimator.enhanced_data.shape}")
        print(f"Rate range: ${estimator.enhanced_data['rate'].min():.2f} - ${estimator.enhanced_data['rate'].max():.2f}")
        print(f"Rate mean: ${estimator.enhanced_data['rate'].mean():.2f}")
        print(f"Rate std: ${estimator.enhanced_data['rate'].std():.2f}")
        
        # Check feature matrix
        print(f"\n🔧 FEATURE MATRIX:")
        print(f"Feature matrix shape: {estimator.feature_matrix.shape}")
        print(f"Feature names count: {len(estimator.feature_names)}")
        
        # Check for any extreme values in features
        feature_extremes = []
        for i, name in enumerate(estimator.feature_names):
            col_data = estimator.feature_matrix[:, i]
            if np.abs(col_data).max() > 1000000:  # Very large values
                feature_extremes.append((name, col_data.min(), col_data.max()))
        
        if feature_extremes:
            print("\n⚠️  EXTREME FEATURE VALUES DETECTED:")
            for name, min_val, max_val in feature_extremes:
                print(f"   {name}: {min_val:.2f} to {max_val:.2f}")
        
        # Test prediction for zipcode 91030
        print(f"\n🎯 TESTING ZIPCODE 91030:")
        
        # Get coordinates
        zip_coords = estimator.zip_coords.get('91030')
        if zip_coords:
            lat, lng = zip_coords
            print(f"Coordinates: {lat}, {lng}")
            
            # Build prediction features
            features = estimator._build_prediction_features(
                '91030', lat, lng, 'import', 'J.B. Hunt Transport'
            )
            
            print(f"Prediction features length: {len(features)}")
            print(f"Training features length: {len(estimator.feature_names)}")
            
            # Check for extreme values in prediction features
            extreme_pred_features = []
            for i, val in enumerate(features):
                if abs(val) > 1000000:
                    feature_name = estimator.feature_names[i] if i < len(estimator.feature_names) else f"feature_{i}"
                    extreme_pred_features.append((feature_name, val))
            
            if extreme_pred_features:
                print("\n⚠️  EXTREME PREDICTION FEATURE VALUES:")
                for name, val in extreme_pred_features:
                    print(f"   {name}: {val:.2f}")
            
            # Test individual model predictions
            print(f"\n🤖 INDIVIDUAL MODEL PREDICTIONS:")
            for name, model_info in estimator.models.items():
                try:
                    if model_info.get('scaled', False):
                        X_pred = estimator.scalers['standard'].transform([features])
                        pred = model_info['model'].predict(X_pred)[0]
                    else:
                        pred = model_info['model'].predict([features])[0]
                    
                    print(f"   {name}: ${pred:.2f}")
                    
                    # Check if this model is causing the issue
                    if abs(pred) > 10000:
                        print(f"      ⚠️  {name} producing extreme value!")
                        
                except Exception as e:
                    print(f"   {name}: ERROR - {e}")
            
        else:
            print("❌ Could not find coordinates for zipcode 91030")
            
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rate_prediction()
