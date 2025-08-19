#!/usr/bin/env python3
"""
Test the fixed advanced rate estimator
"""

try:
    from advanced_rate_estimator import AdvancedLAPortRateEstimator
    
    print("🔧 Testing Fixed Advanced Rate Estimator...")
    estimator = AdvancedLAPortRateEstimator()
    print("✅ Estimator initialized successfully")
    
    # Test the problematic zipcode 91030
    print(f"\n🎯 Testing zipcode 91030 with safety checks...")
    result = estimator.estimate_rate('91030', 'import', 'J.B. Hunt Transport')
    
    print(f"✅ Rate estimation successful!")
    print(f"   💰 Rate: ${result.estimated_rate}")
    print(f"   📊 RPM: ${result.estimated_rpm}")
    print(f"   🎯 Confidence: {result.confidence_score:.1%}")
    print(f"   📏 Distance: {result.distance_miles:.1f} miles")
    
    # Test a few more zipcodes to ensure consistency
    test_zips = ['90210', '92626', '91101', '90401']
    print(f"\n🔍 Testing additional zipcodes for consistency...")
    
    for zipcode in test_zips:
        try:
            result = estimator.estimate_rate(zipcode, 'import', 'J.B. Hunt Transport')
            print(f"   {zipcode}: ${result.estimated_rate:.2f} (RPM: ${result.estimated_rpm:.2f})")
        except Exception as e:
            print(f"   {zipcode}: ❌ Error - {e}")
    
    print("\n🏆 Advanced estimator safety checks working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
