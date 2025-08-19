#!/usr/bin/env python3
"""
Simple diagnosis of the prediction issue
"""

def diagnose_prediction_issue():
    print("🔍 SIMPLE DIAGNOSIS OF PREDICTION ISSUE")
    print("=" * 50)
    
    try:
        from advanced_rate_estimator import AdvancedLAPortRateEstimator
        
        # Initialize with minimal output
        print("Loading estimator...")
        estimator = AdvancedLAPortRateEstimator()
        
        # Test a simple prediction to see what's happening
        print("\nTesting prediction for zipcode 91030...")
        
        # Get the zipcode coordinates
        zip_coords = estimator.zipcode_coords.get('91030')
        if not zip_coords:
            print("❌ Cannot find coordinates for 91030")
            return
        
        lat, lng = zip_coords
        print(f"✅ Coordinates: {lat}, {lng}")
        
        # Try to build prediction features
        try:
            features = estimator._build_prediction_features(
                '91030', lat, lng, 'import', 'J.B. Hunt Transport'
            )
            print(f"✅ Built {len(features)} prediction features")
            
            # Check for obviously problematic values
            problem_features = [i for i, val in enumerate(features) if abs(val) > 100000]
            if problem_features:
                print(f"⚠️  Found {len(problem_features)} features with extreme values:")
                for i in problem_features[:5]:  # Show first 5
                    if i < len(estimator.feature_names):
                        print(f"   {estimator.feature_names[i]}: {features[i]}")
            
        except Exception as e:
            print(f"❌ Error building features: {e}")
            return
        
        # The core issue might be in the model training data
        # Let's check if training data has reasonable values
        print(f"\nChecking training data...")
        
        if hasattr(estimator, 'enhanced_data'):
            rates = estimator.enhanced_data['rate']
            print(f"✅ Training rates: ${rates.min():.2f} to ${rates.max():.2f}")
            print(f"   Mean: ${rates.mean():.2f}, Std: ${rates.std():.2f}")
            
            # If rates are reasonable but predictions are not, 
            # the issue is in feature engineering or model training
            if rates.min() > 0 and rates.max() < 5000:
                print("✅ Training rates look reasonable")
                print("⚠️  Issue likely in feature engineering or prediction pipeline")
            else:
                print("❌ Training rates look problematic")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_prediction_issue()
