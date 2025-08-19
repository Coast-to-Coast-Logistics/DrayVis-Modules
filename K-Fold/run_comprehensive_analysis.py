#!/usr/bin/env python3
"""
K-FOLD COMPREHENSIVE ANALYSIS RUNNER
====================================

Simple script to run the complete K-Fold validation system with all features:
- Parallel validation of all 9 configurations
- Performance dashboard generation
- Auto-optimization recommendations
- AI-readable structured output

Usage:
    python run_comprehensive_analysis.py

Outputs:
    - reports/charts/performance_dashboard_YYYYMMDD_HHMMSS.png
    - reports/data/validation_results.json
    - reports/reports/optimization_recommendations.txt
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run the comprehensive K-Fold validation analysis"""
    
    print("🚀 K-FOLD COMPREHENSIVE VALIDATION SYSTEM")
    print("=" * 55)
    print("Starting complete validation analysis...")
    print("Features enabled:")
    print("  ✅ Parallel validation of all 9 configurations")
    print("  ✅ Performance dashboard with charts")
    print("  ✅ Business intelligence analysis")
    print("  ✅ Auto-optimization recommendations")
    print("  ✅ AI-readable structured output")
    print("=" * 55)
    print()
    
    try:
        # Import after path setup
        from comprehensive_validation_orchestrator import ComprehensiveValidationOrchestrator
        from auto_optimization_engine import AutoOptimizationEngine
        
        # Initialize the system
        print("🔧 Initializing K-Fold Validation System...")
        orchestrator = ComprehensiveValidationOrchestrator()
        optimizer = AutoOptimizationEngine()
        
        # Run comprehensive analysis
        print("📊 Running comprehensive validation analysis...")
        results = orchestrator.run_comprehensive_analysis(fast_mode=True)
        
        print("🧠 Generating optimization recommendations...")
        optimization_results = optimizer.analyze_and_optimize(results)
        
        print("✅ Analysis complete!")
        print()
        print("📁 Generated outputs:")
        print(f"   📊 Charts: reports/charts/")
        print(f"   📄 Reports: reports/reports/")
        print(f"   📋 Data: reports/data/")
        print()
        print("🎯 Key Results:")
        if 'all_results' in results:
            config_count = len(results['all_results'])
            print(f"   • Validated {config_count} configurations")
            print(f"   • Generated performance dashboard")
            print(f"   • Created optimization recommendations")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required files are in the K-Fold directory:")
        print("  - kfold_validation.py")
        print("  - comprehensive_validation_orchestrator.py") 
        print("  - auto_optimization_engine.py")
        return None
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Change to script directory to ensure relative paths work
    script_dir = Path(__file__).parent
    original_cwd = os.getcwd()
    
    try:
        os.chdir(script_dir)
        results = main()
        
        if results:
            print("🎉 K-Fold validation analysis completed successfully!")
        else:
            print("⚠️ Analysis encountered errors. Check the output above.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
