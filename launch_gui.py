"""
DrayVis GUI Launcher
===================

Simple launcher for the DrayVis GUI Rate Estimator.
Double-click this file to start the application.

Author: DrayVis Analytics Team
Date: August 19, 2025
"""

import sys
import os

def main():
    """Launch the DrayVis GUI"""
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import and run GUI
        from gui_rate_estimator import main as gui_main
        
        print("🚛 Starting DrayVis Rate Estimator...")
        gui_main()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nPlease ensure all required modules are installed:")
        print("• tkinter (usually included with Python)")
        print("• pandas")
        print("• numpy") 
        print("• geopy")
        print("• scikit-learn")
        input("\nPress Enter to exit...")
        
    except Exception as e:
        print(f"❌ Error starting DrayVis: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
