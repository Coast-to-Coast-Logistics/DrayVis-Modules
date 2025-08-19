#!/usr/bin/env python3
"""
Test coordinate lookup specifically
"""

def test_coordinate_lookup():
    print("🔍 TESTING COORDINATE LOOKUP")
    print("=" * 40)
    
    try:
        from advanced_rate_estimator import AdvancedLAPortRateEstimator
        
        estimator = AdvancedLAPortRateEstimator()
        
        # Test coordinate lookup for zipcode 91030
        print("Testing coordinate lookup for 91030...")
        lat, lng = estimator._get_zipcode_coordinates('91030')
        
        if lat is not None and lng is not None:
            print(f"✅ Found coordinates: {lat}, {lng}")
            
            # Now test the full prediction
            print("Testing full prediction...")
            result = estimator.estimate_rate('91030', 'import', 'J.B. Hunt Transport')
            print(f"✅ Prediction: ${result.estimated_rate}")
            
        else:
            print("❌ Could not find coordinates for 91030")
            print("Trying fallback zipcode from training data...")
            
            # Get a zipcode from training data
            training_zips = estimator.enhanced_data['destination_zip'].unique()
            test_zip = str(training_zips[0])
            print(f"Testing with training zipcode: {test_zip}")
            
            lat, lng = estimator._get_zipcode_coordinates(test_zip)
            print(f"Coordinates: {lat}, {lng}")
            
            if lat is not None:
                result = estimator.estimate_rate(test_zip, 'import', 'J.B. Hunt Transport')
                print(f"✅ Prediction: ${result.estimated_rate}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_coordinate_lookup()
