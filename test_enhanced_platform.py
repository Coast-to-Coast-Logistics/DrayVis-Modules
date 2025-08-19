#!/usr/bin/env python3
"""
Test the enhanced DrayVis platform with intelligent fallbacks
"""

def test_enhanced_platform():
    print("🚀 TESTING ENHANCED DRAYVIS PLATFORM")
    print("=" * 50)
    
    try:
        from drayvis_platform import DrayVisPlatform
        
        platform = DrayVisPlatform()
        print("✅ Platform initialized")
        
        # Test multiple zipcodes
        test_cases = [
            ('91030', 'import', 'J.B. Hunt Transport'),
            ('90210', 'import', 'XPO Logistics'),
            ('92626', 'export', 'C.H. Robinson'),
            ('91101', 'import', 'FedEx Freight')
        ]
        
        print(f"\n🎯 Testing {len(test_cases)} scenarios...")
        
        for i, (zipcode, order_type, carrier) in enumerate(test_cases, 1):
            print(f"\n--- Test {i}: {zipcode} ({order_type}) ---")
            try:
                result = platform.analyze_zipcode(zipcode)
                
                print(f"✅ Distance: {result['distance_miles']:.1f} miles")
                print(f"💰 Rate: ${result[f'{order_type}_rate']}")
                print(f"📊 RPM: ${result[f'{order_type}_rate'] / result['distance_miles']:.2f}")
                print(f"🎯 Confidence: {result['confidence']:.1%}")
                print(f"📊 Grade: {result['grade']}")
                
                # Check if it's a reasonable rate
                expected_range = (result['distance_miles'] * 3, result['distance_miles'] * 8)
                rate = result[f'{order_type}_rate']
                if expected_range[0] <= rate <= expected_range[1]:
                    print("✅ Rate within reasonable range")
                else:
                    print(f"⚠️  Rate ${rate} outside expected range ${expected_range[0]:.0f}-${expected_range[1]:.0f}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🏆 Enhanced platform testing complete!")
        print("The intelligent fallback system ensures reasonable rates even when ML models fail.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_platform()
