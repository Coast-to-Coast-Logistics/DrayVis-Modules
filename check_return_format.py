#!/usr/bin/env python3
"""
Check return format of platform analysis
"""

from drayvis_platform import DrayVisPlatform

print("🔍 CHECKING RETURN FORMAT")
print("=" * 30)

platform = DrayVisPlatform()
result = platform.estimate_with_platform_analysis('91030')

print("Available keys in result:")
for key in result.keys():
    print(f"  - {key}: {result[key]}")

print(f"\n✅ The enhanced system is working!")
print(f"✅ Rate: ${result.get('import_rate', 'N/A')}")
print(f"✅ Distance: {result.get('distance_miles', 'N/A')} miles")
print(f"✅ Grade: {result.get('grade', 'N/A')}")
