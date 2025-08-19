#!/usr/bin/env python3
"""
🚛 DrayVis Rate Estimation Platform
===================================

Professional-grade rate estimation platform for LA port drayage operations.
Provides accurate rate estimates for unused zipcodes with platform-grade validation.

Platform Standards:
- EXCELLENT: Within 5% of market rates (Platform Grade)
- GOOD: Within 10% of market rates (Acceptable)
- FAIR: 10-15% variance (Needs Review)
- POOR: >15% variance (Requires Calibration)
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from datetime import datetime
try:
    from advanced_rate_estimator import AdvancedLAPortRateEstimator as LAPortRateEstimator
    print("🎯 Using Advanced Rate Estimator v2.0")
except ImportError:
    from intelligent_rate_estimator import LAPortRateEstimator
    print("⚠️  Using fallback rate estimator")

class DrayVisPlatform:
    """Professional rate estimation platform for LA port drayage"""
    
    def __init__(self):
        self.version = "2.0.0"
        self.platform_name = "DrayVis Advanced Rate Estimation Platform"
        
        print(f"🚛 {self.platform_name} v{self.version}")
        print("=" * 60)
        print("Professional-grade rate estimation with advanced ML ensemble")
        print("Enhanced accuracy targeting ±2-3% variance (vs ±5% standard)\n")
        
        # Initialize core components
        self.estimator = LAPortRateEstimator()
        self.zipcode_coords = pd.read_csv("data/us_zip_coordinates.csv")
        self.zipcode_coords['ZIP'] = self.zipcode_coords['ZIP'].astype(str).str.zfill(5)
        
        # Load system data
        self.used_zipcodes = self._get_used_zipcodes()
        self.la_region_unused_zips = self._get_la_region_unused_zipcodes()
        
        # Platform statistics
        self.total_zipcodes_available = len(self.zipcode_coords)
        self.training_zipcodes = len(self.used_zipcodes)
        self.testable_zipcodes = len(self.la_region_unused_zips)
        
        print(f"📊 Platform Statistics:")
        print(f"   • Total US Zipcodes: {self.total_zipcodes_available:,}")
        print(f"   • Training Zipcodes: {self.training_zipcodes:,}")
        print(f"   • Testable LA Region: {self.testable_zipcodes:,}")
        print(f"   • Platform Coverage: {(self.testable_zipcodes/self.training_zipcodes)*100:.1f}% additional capacity")
        
        # Performance tracking
        self.session_tests = []
        self.session_start = datetime.now()
        
    def _get_used_zipcodes(self) -> set:
        """Extract all zipcodes used in training data"""
        dummy_data = pd.read_csv("data/port_drayage_dummy_data.csv")
        used_zips = set()
        used_zips.update(dummy_data['origin_zip'].astype(str).str.zfill(5))
        used_zips.update(dummy_data['destination_zip'].astype(str).str.zfill(5))
        return used_zips
    
    def _calculate_distance_to_port(self, lat: float, lng: float) -> float:
        """Calculate distance from coordinates to LA port"""
        port_lat, port_lng = 33.745762, -118.208042
        R = 3959  # Earth's radius in miles
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat, lng, port_lat, port_lng])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _get_la_region_unused_zipcodes(self) -> List[Dict]:
        """Get LA region zipcodes not used in training data"""
        la_region_zips = []
        
        for _, row in self.zipcode_coords.iterrows():
            zipcode = str(row['ZIP']).zfill(5)
            lat, lng = row['LAT'], row['LNG']
            
            if zipcode in self.used_zipcodes:
                continue
                
            distance = self._calculate_distance_to_port(lat, lng)
            
            if distance <= 200:  # LA region only
                city_state = self._get_city_state_approximation(lat, lng)
                la_region_zips.append({
                    'zipcode': zipcode,
                    'lat': lat,
                    'lng': lng,
                    'distance': distance,
                    'city_state': city_state
                })
        
        la_region_zips.sort(key=lambda x: x['distance'])
        return la_region_zips
    
    def _get_city_state_approximation(self, lat: float, lng: float) -> str:
        """Approximate city/state based on coordinates"""
        if 33.5 <= lat <= 34.5 and -119.0 <= lng <= -117.0:
            if lat >= 34.0:
                return "Los Angeles Metro, CA"
            else:
                return "Orange County, CA"
        elif 32.5 <= lat <= 33.5 and -117.5 <= lng <= -116.5:
            return "San Diego County, CA"
        elif 34.0 <= lat <= 35.0 and -118.5 <= lng <= -117.0:
            return "Inland Empire, CA"
        elif 34.5 <= lat <= 36.0 and -120.0 <= lng <= -118.0:
            return "Central Valley, CA"
        else:
            return "Southern California"
    
    def _get_similar_distance_rates(self, target_distance: float, tolerance: float = 15.0) -> List[Dict]:
        """Get rates for similar distance zipcodes from training data"""
        dummy_data = pd.read_csv("data/port_drayage_dummy_data.csv")
        
        similar_distances = dummy_data[
            abs(dummy_data['miles'] - target_distance) <= tolerance
        ].copy()
        
        if len(similar_distances) == 0:
            tolerance = 25.0
            similar_distances = dummy_data[
                abs(dummy_data['miles'] - target_distance) <= tolerance
            ].copy()
        
        similar_distances['RPM_calculated'] = similar_distances['rate'] / similar_distances['miles']
        
        similar_rates = []
        for _, row in similar_distances.iterrows():
            target_zip = row['destination_zip'] if row['order_type'] == 'import' else row['origin_zip']
            similar_rates.append({
                'zipcode': str(target_zip).zfill(5),
                'distance': row['miles'],
                'rate': row['rate'],
                'rpm': row['RPM_calculated'],
                'carrier': row['carrier'],
                'order_type': row['order_type']
            })
        
        similar_rates.sort(key=lambda x: abs(x['distance'] - target_distance))
        return similar_rates[:5]
    
    def _calculate_platform_grade(self, rate_diff_pct: float) -> Dict[str, str]:
        """Calculate platform grade based on enhanced accuracy standards"""
        if abs(rate_diff_pct) < 3:
            return {"grade": "EXCEPTIONAL", "symbol": "🌟", "description": "Advanced Grade (±3%)"}
        elif abs(rate_diff_pct) < 5:
            return {"grade": "EXCELLENT", "symbol": "🏆", "description": "Platform Grade (±5%)"}
        elif abs(rate_diff_pct) < 8:
            return {"grade": "GOOD", "symbol": "✅", "description": "Acceptable (±8%)"}
        elif abs(rate_diff_pct) < 15:
            return {"grade": "FAIR", "symbol": "⚠️", "description": "Needs Review (±15%)"}
        else:
            return {"grade": "POOR", "symbol": "🚨", "description": "Requires Calibration"}
    
    def estimate_with_platform_analysis(self, zipcode: str, order_type: str = 'import', 
                                      carrier: str = 'J.B. Hunt Transport') -> Dict:
        """Generate comprehensive platform-grade rate analysis"""
        
        print(f"\n{'='*80}")
        print(f"🏭 DRAYVIS PLATFORM ANALYSIS - ZIPCODE {zipcode}")
        print(f"{'='*80}")
        
        # Get zipcode info
        zip_info = None
        for zip_data in self.la_region_unused_zips:
            if zip_data['zipcode'] == zipcode:
                zip_info = zip_data
                break
        
        if not zip_info:
            print(f"❌ Error: Zipcode {zipcode} not available in LA region testable zipcodes")
            return {}
        
        # Generate estimate
        try:
            estimate = self.estimator.estimate_rate(zipcode, order_type, carrier)
        except Exception as e:
            print(f"❌ Platform Error: {e}")
            return {}
        
        # Basic information
        print(f"📍 Service Area: {zip_info['city_state']}")
        print(f"📍 Coordinates: {zip_info['lat']:.6f}, {zip_info['lng']:.6f}")
        print(f"📏 Distance: {estimate.distance_miles:.2f} miles from LA Port")
        print(f"📦 Service Type: {order_type.title()}")
        print(f"🚛 Carrier: {carrier}")
        
        # Primary estimate
        print(f"\n💰 PLATFORM ESTIMATE:")
        print(f"   Rate: ${estimate.estimated_rate:.2f}")
        print(f"   RPM: ${estimate.estimated_rpm:.2f}")
        print(f"   Confidence: {estimate.confidence_score:.1%}")
        
        # Enhanced prediction details
        if hasattr(estimate, 'prediction_details') and estimate.prediction_details:
            details = estimate.prediction_details
            if 'ensemble_size' in details:
                print(f"   🤖 Ensemble Models: {details['ensemble_size']}")
                if 'prediction_std' in details:
                    print(f"   📊 Prediction Std: ±${details['prediction_std']:.2f}")
        
        # Market intelligence
        if hasattr(estimate, 'market_intelligence') and estimate.market_intelligence:
            market = estimate.market_intelligence
            print(f"   🏪 Market Segment: {market.get('market_segment', 'unknown').title()}")
            if 'carrier_efficiency' in market:
                print(f"   🚛 Carrier Efficiency: {market['carrier_efficiency']:.2f}")
        
        # Platform validation
        similar_rates = self._get_similar_distance_rates(estimate.distance_miles)
        
        if similar_rates:
            similar_rates_values = [r['rate'] for r in similar_rates]
            avg_similar_rate = np.mean(similar_rates_values)
            rate_difference = estimate.estimated_rate - avg_similar_rate
            rate_diff_pct = (rate_difference / avg_similar_rate) * 100
            
            grade_info = self._calculate_platform_grade(rate_diff_pct)
            
            print(f"\n🎯 PLATFORM VALIDATION:")
            print(f"   Market Average: ${avg_similar_rate:.2f}")
            print(f"   Our Estimate: ${estimate.estimated_rate:.2f}")
            print(f"   Variance: {rate_diff_pct:+.1f}%")
            print(f"   Grade: {grade_info['symbol']} {grade_info['grade']} - {grade_info['description']}")
            
            # Market comparison details
            print(f"\n📊 MARKET COMPARISON ({len(similar_rates)} similar routes):")
            for i, rate_data in enumerate(similar_rates, 1):
                variance = ((rate_data['rate'] - avg_similar_rate) / avg_similar_rate) * 100
                print(f"   {i}. {rate_data['zipcode']} ({rate_data['distance']:.1f} mi): ${rate_data['rate']:.2f} ({variance:+.1f}%)")
            
            min_rate, max_rate = min(similar_rates_values), max(similar_rates_values)
            spread = ((max_rate - min_rate) / avg_similar_rate) * 100
            print(f"   Range: ${min_rate:.2f} - ${max_rate:.2f} (±{spread:.1f}% spread)")
            
        else:
            print(f"\n⚠️  LIMITED VALIDATION: No similar distance routes found")
            grade_info = {"grade": "UNVALIDATED", "symbol": "❓", "description": "Insufficient Market Data"}
        
        # Calculation breakdown
        print(f"\n🔧 CALCULATION BREAKDOWN:")
        
        # Enhanced breakdown for advanced estimator
        if hasattr(estimate, 'prediction_details') and estimate.prediction_details:
            details = estimate.prediction_details
            
            if 'individual_predictions' in details and len(details['individual_predictions']) > 1:
                predictions = details['individual_predictions']
                print(f"   🤖 Ensemble Predictions:")
                model_names = ['RandomForest', 'XGBoost', 'GradientBoost', 'ElasticNet', 'Ridge']
                for i, pred in enumerate(predictions[:5]):
                    model_name = model_names[i] if i < len(model_names) else f'Model_{i+1}'
                    print(f"      {model_name}: ${pred:.2f}")
                
                if 'weights' in details:
                    print(f"   ⚖️  Weighted Average: ${np.average(predictions, weights=details['weights']):.2f}")
            
            if details.get('method') == 'fallback':
                print(f"   ⚠️  Using fallback estimation method")
        
        # Show market adjustments
        if hasattr(estimate, 'factors_applied'):
            factors = estimate.factors_applied
            base_rate = estimate.estimated_rate / factors.get('total_adjustment', 1.0)
            print(f"   📊 Base Model: ${base_rate:.2f}")
            
            for factor_name, factor_value in factors.items():
                if factor_name != 'total_adjustment' and factor_name != 'ensemble_model':
                    adjustment_pct = (factor_value - 1.0) * 100
                    if abs(adjustment_pct) > 0.1:
                        symbol = "+" if adjustment_pct > 0 else ""
                        print(f"   {factor_name.replace('_', ' ').title()}: {symbol}{adjustment_pct:.1f}%")
            
            total_adjustment = factors.get('total_adjustment', 1.0)
            print(f"   🎯 Total Adjustment: {((total_adjustment - 1.0) * 100):+.1f}%")
            print(f"   💰 Final Rate: ${base_rate:.2f} × {total_adjustment:.3f} = ${estimate.estimated_rate:.2f}")
        else:
            # Fallback to original calculation breakdown
            base_rate = estimate.estimated_rate / np.prod(list(estimate.factors_applied.values()))
            print(f"   Base Model: ${base_rate:.2f}")
            
            for factor_name, factor_value in estimate.factors_applied.items():
                adjustment_pct = (factor_value - 1.0) * 100
                if abs(adjustment_pct) > 0.1:
                    symbol = "+" if adjustment_pct > 0 else ""
                    print(f"   {factor_name.replace('_', ' ').title()}: {symbol}{adjustment_pct:.1f}%")
            
            cumulative_factor = np.prod(list(estimate.factors_applied.values()))
            print(f"   Total Adjustment: {((cumulative_factor - 1.0) * 100):+.1f}%")
            print(f"   Final Rate: ${base_rate:.2f} × {cumulative_factor:.3f} = ${estimate.estimated_rate:.2f}")
        
        # Session tracking
        test_result = {
            'zipcode': zipcode,
            'estimate': estimate.estimated_rate,
            'grade': grade_info['grade'],
            'variance': rate_diff_pct if similar_rates else None,
            'timestamp': datetime.now()
        }
        self.session_tests.append(test_result)
        
        return test_result
    
    def show_available_zipcodes(self, limit: int = 25) -> None:
        """Display available test zipcodes by category"""
        print(f"\n📋 AVAILABLE TEST ZIPCODES (Showing {limit}):")
        print(f"{'='*80}")
        
        categories = {
            'Urban Core': [z for z in self.la_region_unused_zips if z['distance'] < 25],
            'Metro Area': [z for z in self.la_region_unused_zips if 25 <= z['distance'] < 50],
            'Extended LA': [z for z in self.la_region_unused_zips if 50 <= z['distance'] < 100],
            'Regional': [z for z in self.la_region_unused_zips if 100 <= z['distance'] <= 200]
        }
        
        shown = 0
        for category, zips in categories.items():
            if shown >= limit:
                break
            print(f"\n🏪 {category} ({len(zips)} available):")
            category_limit = min(len(zips), limit - shown, 8)
            for i, zip_data in enumerate(zips[:category_limit]):
                print(f"   {zip_data['zipcode']} - {zip_data['distance']:.1f} mi - {zip_data['city_state']}")
                shown += 1
        
        if shown < len(self.la_region_unused_zips):
            remaining = len(self.la_region_unused_zips) - shown
            print(f"\n... and {remaining:,} more zipcodes available for testing")
    
    def run_accuracy_benchmark(self, sample_size: int = 10) -> None:
        """Run accuracy benchmark across distance ranges"""
        print(f"\n🎯 PLATFORM ACCURACY BENCHMARK ({sample_size} samples)")
        print(f"{'='*80}")
        
        # Select diverse samples
        test_samples = []
        ranges = [(0, 25), (25, 50), (50, 100), (100, 200)]
        samples_per_range = sample_size // len(ranges)
        
        for min_dist, max_dist in ranges:
            range_zips = [z for z in self.la_region_unused_zips 
                         if min_dist <= z['distance'] < max_dist]
            if range_zips:
                test_samples.extend(range_zips[:samples_per_range])
        
        # Fill remainder
        remaining = sample_size - len(test_samples)
        if remaining > 0:
            available = [z for z in self.la_region_unused_zips if z not in test_samples]
            test_samples.extend(available[:remaining])
        
        results = []
        excellent_count = 0
        
        for i, zip_data in enumerate(test_samples, 1):
            print(f"\n📊 Benchmark Test {i}/{len(test_samples)}: {zip_data['zipcode']}")
            result = self.estimate_with_platform_analysis(zip_data['zipcode'])
            
            if result.get('grade') == 'EXCELLENT':
                excellent_count += 1
            
            results.append(result)
        
        # Benchmark summary
        print(f"\n🏆 BENCHMARK RESULTS:")
        print(f"{'='*50}")
        
        grade_counts = {}
        valid_variances = []
        
        for result in results:
            grade = result.get('grade', 'UNKNOWN')
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            if result.get('variance') is not None:
                valid_variances.append(abs(result['variance']))
        
        for grade, count in grade_counts.items():
            percentage = (count / len(results)) * 100
            print(f"   {grade}: {count}/{len(results)} ({percentage:.1f}%)")
        
        if valid_variances:
            avg_variance = np.mean(valid_variances)
            print(f"\n📈 Platform Performance:")
            print(f"   Average Accuracy: ±{avg_variance:.1f}%")
            print(f"   Platform Grade Rate: {(excellent_count/len(results))*100:.1f}%")
            
            if avg_variance < 3:
                print(f"   🌟 PLATFORM STATUS: EXCEPTIONAL")
            elif avg_variance < 5:
                print(f"   🏆 PLATFORM STATUS: EXCELLENT")
            elif avg_variance < 8:
                print(f"   ✅ PLATFORM STATUS: GOOD")
            else:
                print(f"   ⚠️  PLATFORM STATUS: NEEDS CALIBRATION")
    
    def show_session_summary(self) -> None:
        """Display current session summary"""
        if not self.session_tests:
            print("\n📊 No tests performed this session")
            return
        
        print(f"\n📊 SESSION SUMMARY:")
        print(f"{'='*40}")
        
        session_duration = datetime.now() - self.session_start
        print(f"Session Duration: {session_duration}")
        print(f"Tests Performed: {len(self.session_tests)}")
        
        grade_counts = {}
        valid_variances = []
        
        for test in self.session_tests:
            grade = test.get('grade', 'UNKNOWN')
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            if test.get('variance') is not None:
                valid_variances.append(abs(test['variance']))
        
        for grade, count in grade_counts.items():
            print(f"{grade}: {count}")
        
        if valid_variances:
            avg_variance = np.mean(valid_variances)
            print(f"Average Accuracy: ±{avg_variance:.1f}%")
    
    def run_interactive_platform(self):
        """Run the interactive platform interface"""
        print(f"\n🚀 DRAYVIS PLATFORM - INTERACTIVE MODE")
        print(f"{'='*60}")
        
        while True:
            print(f"\n📋 PLATFORM COMMANDS:")
            print("  🔍 'list' - Show available test zipcodes")
            print("  🎯 'test <zipcode>' - Analyze specific zipcode")
            print("  🏆 'benchmark <count>' - Run accuracy benchmark")
            print("  📊 'summary' - Show session summary")
            print("  ❓ 'help' - Show detailed help")
            print("  🚪 'quit' - Exit platform")
            
            command = input(f"\n💬 DrayVis> ").strip().lower()
            
            if command == 'quit':
                self.show_session_summary()
                print("\n👋 Thank you for using DrayVis Platform!")
                break
            elif command == 'list':
                self.show_available_zipcodes(30)
            elif command.startswith('test '):
                zipcode = command.split()[1]
                if len(zipcode) == 5 and zipcode.isdigit():
                    self.estimate_with_platform_analysis(zipcode)
                else:
                    print("❌ Please enter a valid 5-digit zipcode")
            elif command.startswith('benchmark'):
                parts = command.split()
                count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
                self.run_accuracy_benchmark(count)
            elif command == 'summary':
                self.show_session_summary()
            elif command == 'help':
                self._show_help()
            else:
                print("❌ Unknown command. Type 'help' for assistance.")
    
    def _show_help(self):
        """Show detailed help information"""
        print(f"\n📚 DRAYVIS ADVANCED PLATFORM HELP:")
        print(f"{'='*50}")
        print(f"🎯 Enhanced Platform Standards:")
        print(f"   🌟 EXCEPTIONAL: ±3% variance (Advanced Grade)")
        print(f"   🏆 EXCELLENT: ±5% variance (Platform Grade)")
        print(f"   ✅ GOOD: ±8% variance (Acceptable)")
        print(f"   ⚠️  FAIR: ±15% variance (Needs Review)")
        print(f"   🚨 POOR: >15% variance (Requires Calibration)")
        print(f"\n🤖 Advanced Features:")
        print(f"   • Ensemble ML models (RandomForest, XGBoost, etc.)")
        print(f"   • 40+ advanced features vs basic 8")
        print(f"   • Enhanced market intelligence")
        print(f"   • Carrier profiling system")
        print(f"   • Temporal pattern recognition")
        print(f"   • Geographic market segmentation")
        print(f"\n📊 Available Data:")
        print(f"   • {self.testable_zipcodes:,} LA region zipcodes")
        print(f"   • {self.training_zipcodes:,} training data points")
        print(f"   • 200-mile radius coverage")
        print(f"\n🔍 Commands:")
        print(f"   • list: Browse available zipcodes by category")
        print(f"   • test <zipcode>: Full analysis with market validation")
        print(f"   • benchmark <count>: Test platform accuracy")
        print(f"   • summary: View session performance")

def main():
    """Launch the DrayVis Platform"""
    platform = DrayVisPlatform()
    platform.run_interactive_platform()

if __name__ == "__main__":
    main()
