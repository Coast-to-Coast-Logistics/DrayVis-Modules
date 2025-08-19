#!/usr/bin/env python3
"""
LA Port Region Rate Estimator for Port Drayage Operations

⚠️  GEOGRAPHIC SCOPE LIMITATION:
This system is trained specifically on LA/Long Beach port data (hub: 90802) and 
is only accurate for estimating rates within the Southern California region.

Confidence Zones:
- HIGH (0.8-1.0): Within 200 miles of LA ports (SoCal, Las Vegas, Phoenix)
- MEDIUM (0.6-0.8): 200-400 miles (Central CA, Nevada, Arizona)  
- LOW (0.3-0.6): 400+ miles (accuracy not guaranteed - use with caution)

Key Features:
- LA port-specific distance curves and market factors
- Geographic boundary checking with confidence zones
- Regional market density adjustments for SoCal
- Clear warnings for out-of-region estimates
- Carrier-specific pricing patterns for LA market
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RateEstimate:
    """Container for rate estimation results"""
    zipcode: str
    estimated_rate: float
    estimated_rpm: float
    distance_miles: float
    confidence_score: float
    factors_applied: Dict[str, float]
    similar_zipcodes: List[str]
    explanation: str
    regional_warning: Optional[str] = None

class LAPortRateEstimator:
    """
    Rate estimation system for LA/Long Beach port drayage operations.
    
    ⚠️  REGIONAL LIMITATION: This estimator is trained specifically on 
    LA port data and should only be used for Southern California estimates.
    """
    
    def __init__(self, dummy_data_path: str = "data/port_drayage_dummy_data.csv", 
                 zipcode_data_path: str = "data/us_zip_coordinates.csv"):
        """Initialize the LA port rate estimation system"""
        
        # LA/Long Beach port hub information
        self.port_hub_zip = 90802
        self.port_coordinates = (33.745762, -118.208042)  # Long Beach port
        
        # Geographic confidence zones based on distance from LA port
        self.confidence_zones = {
            'high': {'max_distance': 200, 'confidence_floor': 0.8},    # SoCal region
            'medium': {'max_distance': 400, 'confidence_floor': 0.6},  # Adjacent regions
            'low': {'max_distance': float('inf'), 'confidence_floor': 0.3}  # Distant regions
        }
        self.dummy_data_path = dummy_data_path
        self.zipcode_data_path = zipcode_data_path
        self.port_zipcode = "90802"
        self.port_lat = 33.745762
        self.port_lng = -118.208042
        
        # Load and analyze data
        self._load_data()
        self._analyze_rate_patterns()
        self._build_models()
        
    def _load_data(self):
        """Load dummy data and zipcode coordinates"""
        print("Loading existing rate data and zipcode coordinates...")
        
        # Load dummy data
        self.dummy_data = pd.read_csv(self.dummy_data_path)
        print(f"Loaded {len(self.dummy_data)} existing rate records")
        
        # Load zipcode coordinates
        self.zipcode_coords = pd.read_csv(self.zipcode_data_path)
        self.zipcode_coords['ZIP'] = self.zipcode_coords['ZIP'].astype(str).str.zfill(5)
        self.zipcode_dict = dict(zip(self.zipcode_coords['ZIP'], 
                                   zip(self.zipcode_coords['LAT'], self.zipcode_coords['LNG'])))
        print(f"Loaded coordinates for {len(self.zipcode_coords)} US zipcodes")
        
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points"""
        R = 3959  # Earth's radius in miles
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
        
    def _analyze_rate_patterns(self):
        """Analyze existing data to understand rate patterns"""
        print("Analyzing rate patterns from existing data...")
        
        # Distance-based analysis
        self.dummy_data['distance_band'] = pd.cut(self.dummy_data['miles'], 
                                                 bins=[0, 25, 50, 75, 100, 150, 200, 300], 
                                                 labels=['0-25', '25-50', '50-75', '75-100', 
                                                        '100-150', '150-200', '200-300'])
        
        # Rate patterns by distance
        self.rate_by_distance = self.dummy_data.groupby('distance_band').agg({
            'rate': ['mean', 'std', 'min', 'max'],
            'RPM': ['mean', 'std', 'min', 'max'],
            'miles': 'mean'
        }).round(2)
        
        # Carrier multipliers
        carrier_stats = self.dummy_data.groupby('carrier').agg({
            'rate': 'mean',
            'RPM': 'mean',
            'miles': 'mean'
        })
        overall_avg_rate = self.dummy_data['rate'].mean()
        overall_avg_rpm = self.dummy_data['RPM'].mean()
        
        self.carrier_multipliers = {
            carrier: {
                'rate_multiplier': carrier_stats.loc[carrier, 'rate'] / overall_avg_rate,
                'rpm_multiplier': carrier_stats.loc[carrier, 'RPM'] / overall_avg_rpm,
                'avg_distance': carrier_stats.loc[carrier, 'miles']
            }
            for carrier in carrier_stats.index
        }
        
        # Import vs Export patterns
        self.order_type_patterns = self.dummy_data.groupby('order_type').agg({
            'rate': 'mean',
            'RPM': 'mean',
            'miles': 'mean'
        })
        
        # Create zipcode feature matrix for clustering
        zipcode_features = []
        zipcode_labels = []
        
        for _, row in self.dummy_data.iterrows():
            # Use destination for imports, origin for exports
            target_zip = row['destination_zip'] if row['order_type'] == 'import' else row['origin_zip']
            if target_zip != self.port_zipcode and str(target_zip) in self.zipcode_dict:
                lat, lng = self.zipcode_dict[str(target_zip)]
                zipcode_features.append([lat, lng, row['miles'], row['rate'], row['RPM']])
                zipcode_labels.append(target_zip)
        
        self.zipcode_features = np.array(zipcode_features)
        self.zipcode_labels = np.array(zipcode_labels)
        
        # Build KNN model for finding similar zipcodes
        if len(self.zipcode_features) > 0:
            self.scaler = StandardScaler()
            self.zipcode_features_scaled = self.scaler.fit_transform(self.zipcode_features[:, :3])  # lat, lng, miles
            self.knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
            self.knn_model.fit(self.zipcode_features_scaled)
        
    def _build_models(self):
        """Build predictive models for rate estimation"""
        print("Building predictive models...")
        
        # Prepare features for modeling
        features = []
        targets_rate = []
        targets_rpm = []
        
        for _, row in self.dummy_data.iterrows():
            target_zip = row['destination_zip'] if row['order_type'] == 'import' else row['origin_zip']
            if target_zip != self.port_zipcode and str(target_zip) in self.zipcode_dict:
                lat, lng = self.zipcode_dict[str(target_zip)]
                
                # Feature engineering
                distance = row['miles']
                distance_squared = distance ** 2
                distance_log = math.log(distance + 1)
                lat_normalized = (lat - self.port_lat) ** 2
                lng_normalized = (lng - self.port_lng) ** 2
                
                # Economic factors (simplified - could be enhanced with real economic data)
                urban_factor = 1.0 if distance < 50 else 0.5  # Urban proximity
                coastal_factor = 1.0 if abs(lng + 118) < 2 else 0.8  # Coastal proximity
                
                features.append([
                    distance, distance_squared, distance_log,
                    lat_normalized, lng_normalized,
                    urban_factor, coastal_factor,
                    1.0 if row['order_type'] == 'import' else 0.0  # Order type
                ])
                
                targets_rate.append(row['rate'])
                targets_rpm.append(row['RPM'])
        
        if len(features) == 0:
            print("Warning: No matching zipcodes found for modeling. Using all data instead.")
            # Fallback: use all data with minimal feature engineering
            for _, row in self.dummy_data.iterrows():
                distance = row['miles']
                features.append([
                    distance, distance ** 2, math.log(distance + 1),
                    0.0, 0.0, 1.0, 1.0,  # Default factors
                    1.0 if row['order_type'] == 'import' else 0.0
                ])
                targets_rate.append(row['rate'])
                targets_rpm.append(row['RPM'])
        
        self.model_features = np.array(features)
        self.model_targets_rate = np.array(targets_rate)
        self.model_targets_rpm = np.array(targets_rpm)
        
        if len(self.model_features) > 0:
            # Train models
            self.rate_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            self.rpm_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            
            self.rate_model.fit(self.model_features, self.model_targets_rate)
            self.rpm_model.fit(self.model_features, self.model_targets_rpm)
            
            print(f"Models trained on {len(features)} data points")
        else:
            print("Warning: No data available for training models. Will use distance-based estimation.")
        
    def _get_zipcode_coordinates(self, zipcode: str) -> Tuple[Optional[float], Optional[float]]:
        """Get coordinates for a zipcode"""
        zipcode = str(zipcode).zfill(5)
        if zipcode in self.zipcode_dict:
            return self.zipcode_dict[zipcode]
        return None, None
        
    def _calculate_distance_to_port(self, lat: float, lng: float) -> float:
        """Calculate distance from coordinates to port"""
        return self._haversine_distance(lat, lng, self.port_lat, self.port_lng)
        
    def _find_similar_zipcodes(self, lat: float, lng: float, miles: float) -> List[str]:
        """Find similar zipcodes using KNN"""
        if not hasattr(self, 'knn_model'):
            return []
            
        query_point = self.scaler.transform([[lat, lng, miles]])
        distances, indices = self.knn_model.kneighbors(query_point)
        
        similar_zips = []
        for idx in indices[0]:
            if idx < len(self.zipcode_labels):
                similar_zips.append(self.zipcode_labels[idx])
                
        return similar_zips
        
    def _apply_market_factors(self, base_rate: float, lat: float, lng: float, distance: float) -> Tuple[float, Dict[str, float]]:
        """Apply market-based adjustments to base rate"""
        factors = {}
        adjusted_rate = base_rate
        
        # Distance-based adjustment (economies of scale)
        if distance > 100:
            distance_factor = 0.95  # Slight discount for long distances
        elif distance < 25:
            distance_factor = 1.1   # Premium for short distances (less efficient)
        else:
            distance_factor = 1.0
            
        factors['distance_factor'] = distance_factor
        adjusted_rate *= distance_factor
        
        # Urban density factor
        urban_premium = 1.0
        if distance < 50:  # Close to port, likely urban
            urban_premium = 1.05  # 5% premium for urban congestion
        elif distance > 150:  # Far from port, likely rural
            urban_premium = 1.02  # 2% premium for rural access difficulty
            
        factors['urban_premium'] = urban_premium
        adjusted_rate *= urban_premium
        
        # Geographic complexity (mountains, etc.)
        # Simplified model - could be enhanced with real geographic data
        if lat > 35.5:  # Northern California mountains
            geo_factor = 1.08
        elif abs(lng + 117) < 1 and lat > 34:  # Inland empire mountains
            geo_factor = 1.04
        else:
            geo_factor = 1.0
            
        factors['geographic_factor'] = geo_factor
        adjusted_rate *= geo_factor
        
        return adjusted_rate, factors
        
    def _calculate_confidence_score(self, zipcode: str, distance: float, similar_zips: List[str]) -> float:
        """Calculate confidence score for the estimate"""
        confidence = 0.5  # Base confidence
        
        # Distance confidence (higher for distances similar to training data)
        distance_counts = self.dummy_data['distance_band'].value_counts()
        distance_band = None
        if distance <= 25:
            distance_band = '0-25'
        elif distance <= 50:
            distance_band = '25-50'
        elif distance <= 75:
            distance_band = '50-75'
        elif distance <= 100:
            distance_band = '75-100'
        elif distance <= 150:
            distance_band = '100-150'
        elif distance <= 200:
            distance_band = '150-200'
        else:
            distance_band = '200-300'
            
        if distance_band in distance_counts:
            distance_confidence = min(distance_counts[distance_band] / 100, 0.4)
            confidence += distance_confidence
            
        # Similar zipcodes confidence
        if len(similar_zips) >= 3:
            similarity_confidence = 0.3
        elif len(similar_zips) >= 1:
            similarity_confidence = 0.2
        else:
            similarity_confidence = 0.0
            
        confidence += similarity_confidence
        
        return min(confidence, 1.0)
    
    def _get_regional_confidence_and_warning(self, target_zipcode: str, distance_miles: float) -> Tuple[float, Optional[str]]:
        """
        Determine confidence adjustment and warning based on distance from LA port.
        
        Returns:
            Tuple of (confidence_multiplier, warning_message)
        """
        warning = None
        
        if distance_miles <= self.confidence_zones['high']['max_distance']:
            # High confidence zone - Southern California region
            confidence_multiplier = 1.0
        elif distance_miles <= self.confidence_zones['medium']['max_distance']:
            # Medium confidence zone - Regional adjacent areas
            confidence_multiplier = 0.85
            warning = f"⚠️  Zipcode {target_zipcode} is {distance_miles:.0f} miles from LA port region. Estimate accuracy may be reduced for markets outside Southern California."
        else:
            # Low confidence zone - Far from training region  
            confidence_multiplier = 0.5
            warning = f"🚨 WARNING: Zipcode {target_zipcode} is {distance_miles:.0f} miles from LA port region. This estimator is trained on Southern California data only. Accuracy is NOT guaranteed for distant markets with different pricing dynamics."
        
        return confidence_multiplier, warning
        
    def estimate_rate(self, zipcode: str, order_type: str = 'import', 
                     carrier: str = None, date: str = None) -> RateEstimate:
        """
        Generate rate estimate for a given zipcode
        
        Args:
            zipcode: Target zipcode (5 digits)
            order_type: 'import' or 'export'
            carrier: Carrier name (optional, will use average if not specified)
            date: Date string (optional, for seasonal adjustments)
            
        Returns:
            RateEstimate object with detailed estimate information
        """
        zipcode = str(zipcode).zfill(5)
        
        # Get coordinates
        lat, lng = self._get_zipcode_coordinates(zipcode)
        if lat is None or lng is None:
            raise ValueError(f"Zipcode {zipcode} not found in database")
            
        # Calculate distance
        distance = self._calculate_distance_to_port(lat, lng)
        
        # Find similar zipcodes
        similar_zips = self._find_similar_zipcodes(lat, lng, distance)
        
        # Prepare features for model prediction
        distance_squared = distance ** 2
        distance_log = math.log(distance + 1)
        lat_normalized = (lat - self.port_lat) ** 2
        lng_normalized = (lng - self.port_lng) ** 2
        urban_factor = 1.0 if distance < 50 else 0.5
        coastal_factor = 1.0 if abs(lng + 118) < 2 else 0.8
        order_type_feature = 1.0 if order_type == 'import' else 0.0
        
        model_input = np.array([[
            distance, distance_squared, distance_log,
            lat_normalized, lng_normalized,
            urban_factor, coastal_factor, order_type_feature
        ]])
        
        # Get base predictions
        try:
            if hasattr(self, 'rate_model') and self.rate_model is not None:
                base_rate = self.rate_model.predict(model_input)[0]
                base_rpm = self.rpm_model.predict(model_input)[0]
            else:
                # Fallback to distance-based estimation
                base_rate, base_rpm = self._distance_based_estimation(distance, order_type)
        except Exception as e:
            print(f"Model prediction failed: {e}, using distance-based estimation")
            base_rate, base_rpm = self._distance_based_estimation(distance, order_type)
        
        # Apply market factors
        adjusted_rate, market_factors = self._apply_market_factors(base_rate, lat, lng, distance)
        
        # Apply carrier multiplier if specified
        if carrier and carrier in self.carrier_multipliers:
            carrier_factor = self.carrier_multipliers[carrier]['rate_multiplier']
            adjusted_rate *= carrier_factor
            market_factors['carrier_factor'] = carrier_factor
        else:
            market_factors['carrier_factor'] = 1.0
            
        # Apply order type adjustment
        if order_type in self.order_type_patterns.index:
            order_avg = self.order_type_patterns.loc[order_type, 'rate']
            overall_avg = self.dummy_data['rate'].mean()
            order_factor = order_avg / overall_avg
            adjusted_rate *= order_factor
            market_factors['order_type_factor'] = order_factor
        else:
            market_factors['order_type_factor'] = 1.0
            
        # Calculate final RPM
        final_rpm = adjusted_rate / distance if distance > 0 else base_rpm
        
        # Calculate confidence
        base_confidence = self._calculate_confidence_score(zipcode, distance, similar_zips)
        
        # Apply regional confidence adjustment and get warning
        regional_multiplier, regional_warning = self._get_regional_confidence_and_warning(zipcode, distance)
        final_confidence = base_confidence * regional_multiplier
        
        # Generate explanation
        explanation = self._generate_explanation(zipcode, distance, market_factors, similar_zips)
        
        return RateEstimate(
            zipcode=zipcode,
            estimated_rate=round(adjusted_rate, 2),
            estimated_rpm=round(final_rpm, 2),
            distance_miles=round(distance, 2),
            confidence_score=round(final_confidence, 2),
            factors_applied=market_factors,
            similar_zipcodes=similar_zips[:3],  # Top 3 similar
            explanation=explanation,
            regional_warning=regional_warning
        )
        
    def _generate_explanation(self, zipcode: str, distance: float, factors: Dict[str, float], 
                            similar_zips: List[str]) -> str:
        """Generate human-readable explanation of rate estimate"""
        explanation_parts = []
        
        explanation_parts.append(f"Rate estimate for zipcode {zipcode} ({distance:.1f} miles from port):")
        
        # Distance context
        if distance < 25:
            explanation_parts.append("• Short distance route with urban congestion premium")
        elif distance < 75:
            explanation_parts.append("• Medium distance route in metro area")
        elif distance < 150:
            explanation_parts.append("• Long distance route with economies of scale")
        else:
            explanation_parts.append("• Very long distance route to rural area")
            
        # Factor explanations
        significant_factors = []
        for factor_name, factor_value in factors.items():
            if abs(factor_value - 1.0) > 0.02:  # Only mention significant factors
                if factor_value > 1.0:
                    significant_factors.append(f"{factor_name}: +{(factor_value-1)*100:.1f}%")
                else:
                    significant_factors.append(f"{factor_name}: {(factor_value-1)*100:.1f}%")
                    
        if significant_factors:
            explanation_parts.append("• Adjustments applied: " + ", ".join(significant_factors))
            
        # Similar zipcodes
        if similar_zips:
            similar_zips_str = [str(zip_code) for zip_code in similar_zips[:3]]
            explanation_parts.append(f"• Based on patterns from similar zipcodes: {', '.join(similar_zips_str)}")
            
        return "\n".join(explanation_parts)
    
    def _distance_based_estimation(self, distance, order_type):
        """Fallback distance-based rate estimation"""
        # Base rate calculation using observed patterns from dummy data
        if len(self.dummy_data) > 0:
            avg_rpm = self.dummy_data['RPM'].mean()
            rpm_std = self.dummy_data['RPM'].std()
        else:
            avg_rpm = 2.5  # Default RPM
            rpm_std = 0.5
        
        # Distance-based RPM adjustment
        if distance < 50:
            rpm = avg_rpm + rpm_std * 0.3  # Higher RPM for short distances
        elif distance < 150:
            rpm = avg_rpm
        elif distance < 300:
            rpm = avg_rpm - rpm_std * 0.2
        else:
            rpm = avg_rpm - rpm_std * 0.4
        
        # Order type adjustment
        if order_type == 'export':
            rpm *= 0.95  # Slightly lower for exports
        
        rate = distance * rpm
        
        return rate, rpm
        
    def validate_estimates(self, sample_size: int = 100) -> Dict[str, Any]:
        """Validate estimation accuracy against known data"""
        print(f"Validating rate estimates with {sample_size} random samples...")
        
        # Sample from existing data
        sample_data = self.dummy_data.sample(n=min(sample_size, len(self.dummy_data)))
        
        estimates = []
        actuals = []
        errors = []
        
        for _, row in sample_data.iterrows():
            target_zip = row['destination_zip'] if row['order_type'] == 'import' else row['origin_zip']
            
            if target_zip != self.port_zipcode:
                try:
                    estimate = self.estimate_rate(target_zip, row['order_type'], row['carrier'])
                    estimates.append(estimate.estimated_rate)
                    actuals.append(row['rate'])
                    errors.append(abs(estimate.estimated_rate - row['rate']) / row['rate'] * 100)
                except:
                    continue
                    
        if estimates:
            results = {
                'mean_absolute_error': np.mean([abs(e - a) for e, a in zip(estimates, actuals)]),
                'mean_percentage_error': np.mean(errors),
                'correlation': np.corrcoef(estimates, actuals)[0, 1] if len(estimates) > 1 else 0,
                'sample_size': len(estimates),
                'estimates': estimates[:10],  # First 10 for inspection
                'actuals': actuals[:10]
            }
        else:
            results = {'error': 'No valid estimates generated'}
            
        return results

def main():
    """Demonstration of the LA Port rate estimation system"""
    print("🚛 LA Port Region Rate Estimator")
    print("=" * 50)
    
    # Initialize estimator
    estimator = LAPortRateEstimator()
    
    # Test with some example zipcodes not in the original data
    test_zipcodes = [
        ("10001", "New York, NY"),
        ("60601", "Chicago, IL"), 
        ("30301", "Atlanta, GA"),
        ("98101", "Seattle, WA"),
        ("75201", "Dallas, TX"),
        ("33101", "Miami, FL"),
        ("85001", "Phoenix, AZ"),
        ("94102", "San Francisco, CA")
    ]
    
    print("\n🎯 Rate Estimates for Major US Cities:")
    print("-" * 80)
    
    for zipcode, city in test_zipcodes:
        try:
            # Get both import and export estimates
            import_est = estimator.estimate_rate(zipcode, 'import', 'J.B. Hunt Transport')
            export_est = estimator.estimate_rate(zipcode, 'export', 'J.B. Hunt Transport')
            
            print(f"\n📍 {city} ({zipcode}) - {import_est.distance_miles:.0f} miles from LA Port")
            print(f"   Import Rate: ${import_est.estimated_rate:.2f} (${import_est.estimated_rpm:.2f}/mile)")
            print(f"   Export Rate: ${export_est.estimated_rate:.2f} (${export_est.estimated_rpm:.2f}/mile)")
            print(f"   Confidence: {import_est.confidence_score:.1%}")
            if import_est.similar_zipcodes:
                print(f"   Similar areas: {', '.join(import_est.similar_zipcodes)}")
                
        except Exception as e:
            print(f"\n❌ Could not estimate rate for {city}: {e}")
    
    # Validation
    print(f"\n📊 Model Validation Results:")
    print("-" * 40)
    validation = estimator.validate_estimates(50)
    if 'error' not in validation:
        print(f"Mean Absolute Error: ${validation['mean_absolute_error']:.2f}")
        print(f"Mean Percentage Error: {validation['mean_percentage_error']:.1f}%")
        print(f"Correlation with actual rates: {validation['correlation']:.3f}")
        print(f"Sample size: {validation['sample_size']} estimates")
    else:
        print(f"Validation error: {validation['error']}")

if __name__ == "__main__":
    main()
