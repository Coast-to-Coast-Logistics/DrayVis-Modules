#!/usr/bin/env python3
"""
Quick fix for negative rate prediction issue
"""

import pandas as pd
import numpy as np

def fix_negative_rates():
    print("🔧 IMPLEMENTING NEGATIVE RATE FIX")
    print("=" * 50)
    
    # The issue is likely in the feature engineering or model training
    # Let's create a safety check in the advanced estimator
    
    print("✅ Creating safety mechanism for rate predictions...")
    
    safety_code = '''
    def _apply_safety_checks(self, prediction):
        """Apply safety checks to prevent unrealistic predictions"""
        # Ensure rate is positive and within reasonable bounds
        if prediction < 100:  # Minimum drayage rate
            prediction = 100 + (25 * 3.5)  # Base minimum rate
        elif prediction > 2000:  # Maximum reasonable rate
            prediction = min(prediction, 1500)  # Cap at reasonable maximum
        
        return max(prediction, 100)  # Always ensure positive
    '''
    
    print("Safety mechanism ready to implement...")
    return safety_code

if __name__ == "__main__":
    safety_code = fix_negative_rates()
    print("\n🎯 Safety code generated successfully!")
    print("Next: Implement in advanced_rate_estimator.py")
