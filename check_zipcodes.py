#!/usr/bin/env python3
"""
Check available zipcodes and coordinates
"""

def check_zipcode_data():
    print("🔍 CHECKING ZIPCODE COORDINATE DATA")
    print("=" * 50)
    
    try:
        from advanced_rate_estimator import AdvancedLAPortRateEstimator
        
        estimator = AdvancedLAPortRateEstimator()
        
        # Check what zipcodes are available
        print(f"Available zipcodes: {len(estimator.zipcode_coords)}")
        
        # Show first 10 zipcodes
        zipcode_list = list(estimator.zipcode_coords.keys())[:10]
        print(f"First 10 zipcodes: {zipcode_list}")
        
        # Check if 91030 is in the training data
        if hasattr(estimator, 'enhanced_data'):
            training_zips = estimator.enhanced_data['destination_zip'].unique()
            print(f"\nTraining destination zipcodes: {len(training_zips)}")
            
            if '91030' in training_zips:
                print("✅ 91030 is in training data")
            else:
                print("❌ 91030 NOT in training data")
                print(f"First 10 training zips: {training_zips[:10]}")
        
        # Try a zipcode that should exist
        test_zip = zipcode_list[0] if zipcode_list else None
        if test_zip:
            print(f"\n🎯 Testing with available zipcode: {test_zip}")
            coords = estimator.zipcode_coords[test_zip]
            print(f"Coordinates: {coords}")
            
            try:
                result = estimator.estimate_rate(test_zip, 'import', 'J.B. Hunt Transport')
                print(f"✅ Estimation successful: ${result.estimated_rate}")
            except Exception as e:
                print(f"❌ Estimation failed: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_zipcode_data()
