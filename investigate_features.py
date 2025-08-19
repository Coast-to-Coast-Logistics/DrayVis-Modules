#!/usr/bin/env python3
"""
Investigate and fix the feature mismatch issue
"""

import pandas as pd
import numpy as np
from advanced_rate_estimator import AdvancedLAPortRateEstimator

def investigate_feature_issue():
    print("🔍 INVESTIGATING FEATURE MISMATCH ISSUE")
    print("=" * 60)
    
    try:
        estimator = AdvancedLAPortRateEstimator()
        
        # Check training data range
        print("\n📊 TRAINING DATA ANALYSIS:")
        train_features = estimator.feature_matrix
        print(f"Training features shape: {train_features.shape}")
        
        # Handle numpy array formatting properly
        if hasattr(train_features, 'min'):
            min_val = float(train_features.min())
            max_val = float(train_features.max())
            mean_val = float(train_features.mean())
            std_val = float(train_features.std())
            
            print(f"Feature range: {min_val:.2f} to {max_val:.2f}")
            print(f"Feature mean: {mean_val:.2f}")
            print(f"Feature std: {std_val:.2f}")
        
        # Test prediction features for zipcode 91030
        zip_coords = estimator.zip_coords.get('91030')
        if zip_coords:
            lat, lng = zip_coords
            pred_features = estimator._build_prediction_features(
                '91030', lat, lng, 'import', 'J.B. Hunt Transport'
            )
            
            print(f"\n🎯 PREDICTION FEATURES FOR 91030:")
            print(f"Prediction features length: {len(pred_features)}")
            print(f"Training features length: {train_features.shape[1]}")
            
            if len(pred_features) == train_features.shape[1]:
                print("✅ Feature length match")
                
                # Compare feature ranges
                pred_array = np.array(pred_features)
                print(f"Prediction range: {pred_array.min():.2f} to {pred_array.max():.2f}")
                print(f"Prediction mean: {pred_array.mean():.2f}")
                print(f"Prediction std: {pred_array.std():.2f}")
                
                # Look for extreme values
                extreme_indices = np.where(np.abs(pred_array) > 1000)[0]
                if len(extreme_indices) > 0:
                    print(f"\n⚠️  EXTREME VALUES IN PREDICTION FEATURES:")
                    for idx in extreme_indices:
                        feature_name = estimator.feature_names[idx] if idx < len(estimator.feature_names) else f"feature_{idx}"
                        print(f"   {feature_name}: {pred_array[idx]:.2f}")
                        
                        # Show training range for this feature
                        train_col = train_features[:, idx]
                        print(f"      Training range: {train_col.min():.2f} to {train_col.max():.2f}")
                
            else:
                print("❌ Feature length mismatch!")
                print(f"   Expected: {train_features.shape[1]}")
                print(f"   Got: {len(pred_features)}")
        
        # Test individual model predictions with detailed analysis
        print(f"\n🤖 DETAILED MODEL ANALYSIS:")
        features = pred_features
        
        for name, model_info in estimator.models.items():
            try:
                print(f"\n{name.upper()}:")
                if model_info.get('scaled', False):
                    # Use scaled features
                    X_pred = estimator.scalers['standard'].transform([features])
                    print(f"   Using scaled features: {X_pred[0][:5]}... (showing first 5)")
                    pred = model_info['model'].predict(X_pred)[0]
                else:
                    # Use raw features
                    print(f"   Using raw features: {features[:5]}... (showing first 5)")
                    pred = model_info['model'].predict([features])[0]
                
                print(f"   Prediction: ${pred:.2f}")
                print(f"   CV Score: {model_info['cv_score']:.4f}")
                
            except Exception as e:
                print(f"   ERROR: {e}")
        
    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    investigate_feature_issue()
