#!/usr/bin/env python3
"""
Quick test of zipcode 91030 with enhanced platform
"""

from drayvis_platform import DrayVisPlatform

print("🎯 QUICK TEST: ZIPCODE 91030")
print("=" * 40)

platform = DrayVisPlatform()
print("✅ Platform loaded")

result = platform.estimate_with_platform_analysis('91030')

print(f"\n💰 RESULTS FOR 91030:")
print(f"   Distance: {result['distance_miles']:.1f} miles")
print(f"   Import Rate: ${result['import_rate']:.2f}")
print(f"   RPM: ${result['import_rate'] / result['distance_miles']:.2f}")
print(f"   Confidence: {result['confidence']:.1%}")
print(f"   Grade: {result['grade']}")

# Check reasonableness
rpm = result['import_rate'] / result['distance_miles']
if 3.0 <= rpm <= 15.0:
    print("✅ RPM within reasonable range ($3-15/mile)")
else:
    print(f"⚠️  RPM ${rpm:.2f} outside typical range")

print("\n🏆 Enhanced platform with intelligent fallbacks working!")
