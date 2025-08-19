"""
DrayVis Intelligent Rate Estimator - Enhanced Version
====================================================

A comprehensive system for estimating drayage rates for zip codes with no historical data.
Provides multiple estimation methods with confidence levels and time-weighted accuracy.

Author: DrayVis Analytics Team
Date: August 19, 2025
Version: 2.0 - Enhanced with time weighting and tunable parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from geopy.distance import geodesic
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ========================================================================
# 🎛️ TUNING PARAMETERS - Adjust these for optimal performance
# ========================================================================

class EstimatorConfig:
    """Configuration class for all tuning parameters"""
    
    # ⏰ TIME WEIGHTING PARAMETERS
    TIME_DECAY_DAYS = 90  # Data older than this (days) gets progressively less weight.
    #  - Small (<30): Only very recent data is trusted, old data ignored quickly.
    #  - Large (>180): Old data remains influential, slower adaptation to new
    #    trends.
    #  - Typical: 60-120. Lower for volatile markets, higher for stable ones.
    #  - Effect: Lower values make the estimator react quickly to new trends but
    #    may ignore useful older data. Higher values make the system more stable
    #    but slower to adapt to market changes.
    #  - Technical: Controls the cutoff for time-based exponential decay in
    #    _calculate_time_weight; records older than this are down-weighted more
    #    aggressively, directly affecting the time_weight column and all
    #    time-weighted averages/statistics.

    RECENT_BOOST_DAYS = 30  # Data within this timeframe (days) gets extra boost.
    #  - Small (<15): Only extremely recent data gets a boost.
    #  - Large (>60): Many records get boosted, may overweight short-term
    #    spikes.
    #  - Typical: 20-40. Use higher for fast-changing rates.
    #  - Effect: Higher values make the estimator more sensitive to recent market
    #    spikes, possibly at the expense of stability. Lower values keep the
    #    estimator focused on longer-term trends.
    #  - Technical: Sets the threshold for applying RECENT_MULTIPLIER in
    #    _calculate_time_weight; records newer than this get multiplied,
    #    increasing their influence in all weighted calculations and neighbor
    #    selection.

    TIME_DECAY_RATE = 0.98  # Daily decay factor (0.98 = 1% decay per day).
    #  - Small (<0.90): Old data loses weight very fast (aggressive decay).
    #  - Large (>0.98): Old data loses weight slowly (conservative decay).
    #  - Typical: 0.93-0.97. Lower for volatile, higher for stable.
    #  - Effect: Lower values make the estimator ignore old data quickly,
    #    increasing responsiveness but risking overreaction. Higher values keep
    #    old data relevant, increasing stability but possibly lagging behind new
    #    trends.
    #  - Technical: Used as the base in the exponential decay formula in
    #    _calculate_time_weight; directly determines the rate at which older
    #    records lose influence in all time-weighted calculations.

    RECENT_MULTIPLIER = 1.5  # Extra weight for very recent data.
    #  - Small (<1.2): Recent data barely favored.
    #  - Large (>3): Recent data dominates, old data nearly ignored.
    #  - Typical: 1.5-2.5. Higher for fast-moving markets.
    #  - Effect: Higher values make the estimator prioritize the most recent
    #    transactions, increasing sensitivity to short-term changes. Lower
    #    values balance recent and older data.
    #  - Technical: Multiplies the time_weight for records newer than
    #    RECENT_BOOST_DAYS in _calculate_time_weight, increasing their impact
    #    on all weighted averages, neighbor selection, and confidence scoring.

    # 📍 DISTANCE WEIGHTING PARAMETERS
    KNN_K_NEIGHBORS = 6  # Number of neighbors to consider in KNN.
    #  - Small (<3): Only closest neighbors, may be noisy if data is sparse.
    #  - Large (>12): More neighbors, smoother but less sensitive to local
    #    variation.
    #  - Typical: 5-8. Use higher for dense data, lower for sparse.
    #  - Effect: Higher values smooth out local fluctuations but may dilute
    #    local accuracy. Lower values make the estimator more sensitive to
    #    local conditions but risk noise if data is sparse.
    #  - Technical: Sets the number of zip codes selected in _knn_estimation;
    #    directly controls the size of the neighbor pool for weighted averaging
    #    and confidence calculations.

    MAX_NEIGHBOR_DISTANCE = 15  # Max distance (miles) for neighbors to be considered.
    #  - Small (<5): Only extremely close zips used, may miss useful data.
    #  - Large (>30): Distant zips included, may reduce local accuracy.
    #  - Typical: 8-15. Lower for urban, higher for rural.
    #  - Effect: Higher values allow the estimator to use more distant data,
    #    which can help in sparse areas but may reduce local relevance. Lower
    #    values focus on local data, increasing accuracy in dense regions but
    #    risking lack of data in sparse ones.
    #  - Technical: Used as a cutoff in _knn_estimation; zip codes farther than
    #    this are excluded from neighbor selection, directly affecting the pool
    #    of candidates for estimation.

    DISTANCE_DECAY_FACTOR = 1.3  # Higher = more weight to closer neighbors.
    #  - Small (<1.2): Distant neighbors get almost equal weight.
    #  - Large (>3): Only very close neighbors matter.
    #  - Typical: 1.5-2.5. Higher for urban, lower for rural.
    #  - Effect: Higher values make the estimator focus on the closest neighbors,
    #    increasing local accuracy but risking noise. Lower values spread weight
    #    across more neighbors, smoothing results but possibly reducing local
    #    precision.
    #  - Technical: Used as the exponent in the geographic_weight calculation in
    #    _knn_estimation; higher values make the weight drop off faster with
    #    distance, directly affecting neighbor influence in weighted averaging.

    MIN_DISTANCE_WEIGHT = 0.15  # Minimum weight for furthest neighbors.
    #  - Small (<0.05): Distant neighbors nearly ignored.
    #  - Large (>0.3): Distant neighbors always have some influence.
    #  - Typical: 0.08-0.15. Use higher if data is sparse.
    #  - Effect: Higher values ensure distant neighbors always contribute, which
    #    helps in sparse data situations. Lower values make the estimator ignore
    #    distant neighbors, increasing local focus.
    #  - Technical: Sets the minimum allowed value for geographic_weight in
    #    _knn_estimation, ensuring distant neighbors are not completely excluded
    #    from weighted averaging.

    # 🚢 PORT DISTANCE SIMILARITY PARAMETERS
    PORT_DISTANCE_SIMILARITY_WEIGHT = 5  # How much to weight port distance similarity (multiplier).
    #  - Small (<1): Port distance similarity has little effect.
    #  - Large (>3): Only neighbors with similar port distances matter.
    #  - Typical: 1.5-2.5. Higher for port-centric pricing.
    #  - Effect: Higher values make the estimator focus on neighbors with similar
    #    port distances, increasing accuracy for port-centric pricing. Lower
    #    values allow more diverse neighbors, which may help in non-port-centric
    #    markets.
    #  - Technical: Multiplies the port_distance_weight in _knn_estimation,
    #    amplifying the effect of port distance similarity in neighbor weighting
    #    and selection.

    MAX_PORT_DISTANCE_DIFFERENCE = 2  # Max acceptable port distance difference (miles) for neighbors.
    #  - Small (<2): Only nearly identical port distances allowed.
    #  - Large (>20): Wide range of port distances allowed.
    #  - Typical: 3-10. Lower for strict port-based pricing.
    #  - Effect: Lower values make the estimator only use neighbors with very
    #    similar port distances, increasing port-based accuracy. Higher values             
    #    allow more flexibility but may reduce port-specific precision.
    #  - Technical: Used as a cutoff in _knn_estimation; neighbors with port
    #    distance difference above this get minimum port_distance_weight,
    #    reducing their influence in weighted averaging.

    PORT_DISTANCE_DECAY_FACTOR = 3  # How sharply to penalize port distance differences.
    #  - Small (<1): Port distance difference penalized gently.
    #  - Large (>3): Even small differences penalized heavily.
    #  - Typical: 1.5-2.5. Higher for strict port-based pricing.
    #  - Effect: Higher values make the estimator penalize even small port
    #    distance differences, increasing strictness. Lower values allow more
    #    flexibility in neighbor selection.
    #  - Technical: Used as the exponent in port_distance_weight calculation in
    #    _knn_estimation; higher values make weight drop off faster with port
    #    distance difference, reducing neighbor influence.

    MIN_PORT_DISTANCE_WEIGHT = 0.01  # Minimum weight for very different port distances.
    #  - Small (<0.01): Dissimilar port distances nearly ignored.
    #  - Large (>0.2): Dissimilar port distances always have some influence.
    #  - Typical: 0.03-0.08. Use higher if data is sparse.
    #  - Effect: Higher values ensure even dissimilar port distances contribute,
    #    which helps in sparse data. Lower values make the estimator ignore
    #    dissimilar port distances, increasing port-based focus.
    #  - Technical: Sets the minimum allowed value for port_distance_weight in
    #    _knn_estimation, ensuring neighbors with dissimilar port distances are
    #    not completely excluded from weighted averaging.

    # 🎯 CONFIDENCE LEVEL THRESHOLDS
    CONFIDENCE_VERY_HIGH_MIN = 95  # Minimum for "Very High" confidence (percent).
    #  - Small (<90): "Very High" label given more often.
    #  - Large (>98): Only extremely certain estimates get "Very High".
    #  - Typical: 92-97. Adjust for how strict you want the system to be.
    #  - Effect: Lower values make the system label more estimates as "Very High"
    #    confidence, which may be misleading. Higher values make "Very High"
    #    confidence rare and reserved for only the best estimates.
    #  - Technical: Used as the threshold for assigning "Very High"
    #    confidence_category in _combine_estimates; directly affects how
    #    confidence_level is mapped to output labels.

    CONFIDENCE_HIGH_MIN = 85      # Minimum for "High" confidence.
    #  - Small (<75): "High" label given more often.
    #  - Large (>90): Only very certain estimates get "High".
    #  - Typical: 80-90.
    #  - Effect: Lower values make "High" confidence more common, possibly
    #    overstating certainty. Higher values make "High" confidence more
    #    selective.
    #  - Technical: Used as the threshold for assigning "High"
    #    confidence_category in _combine_estimates; directly affects how
    #    confidence_level is mapped to output labels.

    CONFIDENCE_MEDIUM_MIN = 75    # Minimum for "Medium" confidence.
    #  - Small (<60): "Medium" label given more often.
    #  - Large (>85): Only very certain estimates get "Medium".
    #  - Typical: 70-80.
    # Below MEDIUM_MIN = "Low" confidence.
    #  - Effect: Lower values make "Medium" confidence more common, higher
    #    values make it more selective. Adjust to match your risk tolerance for
    #    estimates.
    #  - Technical: Used as the threshold for assigning "Medium"
    #    confidence_category in _combine_estimates; directly affects how
    #    confidence_level is mapped to output labels.

    # 🔧 MODEL PARAMETERS
    ML_MODEL_CONFIDENCE_BASE = 70  # Base confidence for ML model (percent).
    #  - Small (<50): ML model trusted less.
    #  - Large (>85): ML model trusted more.
    #  - Typical: 65-75.
    #  - Effect: Higher values make the system trust the ML model more,
    #    increasing its influence on the final estimate. Lower values reduce ML
    #    model impact, favoring other methods.
    #  - Technical: Sets the base confidence for ML model predictions in
    #    _distance_model_estimation; directly affects confidence scoring and
    #    method weighting in _combine_estimates.

    ML_EXTRAPOLATION_PENALTY = 1  # Confidence penalty per mile outside training range.
    #  - Small (<1): Extrapolation barely penalized.
    #  - Large (>5): Extrapolation penalized harshly.
    #  - Typical: 1.5-3.5. Higher for volatile markets.
    #  - Effect: Higher values make the system penalize ML model extrapolation
    #    more, reducing confidence for out-of-range predictions. Lower values
    #    allow more trust in extrapolated results.
    #  - Technical: Used in _distance_model_estimation to reduce confidence for
    #    predictions outside the training range; higher values decrease
    #    confidence more rapidly per mile.

    # 📊 RATE VARIANCE PARAMETERS
    CONSISTENCY_BONUS = 20  # Confidence bonus for low variance in neighbors.
    #  - Small (<10): Consistency bonus is minor.
    #  - Large (>30): Consistency bonus is major.
    #  - Typical: 15-25.
    #  - Effect: Higher values make the system reward consistent neighbor rates,
    #    increasing confidence in stable areas. Lower values reduce the impact
    #    of consistency on confidence.
    #  - Technical: Added to consistency_confidence in _calculate_knn_confidence
    #    when neighbor RPM percentage variance is below dynamic threshold; directly
    #    increases confidence_level for stable neighbor pools.

    HIGH_VARIANCE_PENALTY = 15  # Confidence penalty for high percentage variance.
    #  - Small (<5): High percentage variance barely penalized.
    #  - Large (>30): High percentage variance penalized harshly.
    #  - Typical: 10-20.
    #  - Effect: Higher values make the system penalize high relative rate variance more,
    #    reducing confidence in noisy areas. Lower values allow more trust in
    #    variable data.
    #  - Technical: Used to reduce consistency_confidence in
    #    _calculate_knn_confidence when neighbor RPM percentage variance exceeds
    #    dynamic threshold; higher values decrease confidence_level more
    #    rapidly for relative variance.

    VARIANCE_THRESHOLD_BASE_PERCENT = 10  # Base percentage variance threshold (near port).
    #  - Small (<10): Even small percentage variance triggers penalty near port.
    #  - Large (>25): Only very high percentage variance triggers penalty near port.
    #  - Typical: 12-20%. Lower for strict consistency requirements.
    #  - Effect: Lower values make the system penalize relative variance more strictly
    #    near the port where data should be more consistent. Higher values allow
    #    more tolerance even in dense data areas.
    #  - Technical: Used as the base percentage in dynamic variance threshold
    #    calculation in _calculate_knn_confidence; combined with distance scaling
    #    to create adaptive percentage thresholds.

    VARIANCE_DISTANCE_SCALING_PERCENT = 0.05  # Additional percentage variance tolerance per mile from port.
    #  - Small (<0.1): Percentage variance tolerance increases slowly with distance.
    #  - Large (>1.0): Percentage variance tolerance increases rapidly with distance.
    #  - Typical: 0.3-0.8%. Higher for markets with strong distance effects.
    #  - Effect: Higher values make the system more tolerant of relative variance in
    #    remote areas, reflecting natural data sparsity and lower base rates. Lower 
    #    values keep variance standards more uniform across distances.
    #  - Technical: Multiplied by target_port_distance and added to
    #    VARIANCE_THRESHOLD_BASE_PERCENT in _calculate_knn_confidence to create
    #    distance-adaptive percentage variance thresholds.

    # 🏛️ HISTORICAL DATA WEIGHTING
    MIN_HISTORICAL_TRANSACTIONS = 1  # Minimum transactions for "historical" data.
    #  - Small (<1): All zips with any data considered "historical".
    #  - Large (>5): Only zips with many records considered "historical".
    #  - Typical: 1-3.
    #  - Effect: Higher values make the system require more data before trusting
    #    historical rates, increasing reliability but reducing coverage. Lower
    #    values allow more zips to be considered historical, increasing coverage
    #    but risking noise.
    #  - Technical: Used as a cutoff in _get_known_rate and zip_stats; zip codes
    #    with fewer than this number of records are not considered historical,
    #    affecting method selection and confidence scoring.

    HISTORICAL_CONFIDENCE_BOOST = 20  # Extra confidence for zip codes with historical data.
    #  - Small (<5): Historical data barely boosts confidence.
    #  - Large (>20): Historical data strongly boosts confidence.
    #  - Typical: 8-15.
    #  - Effect: Higher values make the system trust historical data more,
    #    increasing confidence for known zips. Lower values reduce the impact of
    #    historical data.
    #  - Technical: Added to confidence_level in _get_known_rate for zip codes
    #    with sufficient historical data, directly increasing output confidence.

    SAMPLE_SIZE_BOOST_FACTOR = 3  # Confidence boost per additional transaction.
    #  - Small (<2): Sample size has little effect.
    #  - Large (>10): Sample size dominates confidence.
    #  - Typical: 3-7.
    #  - Effect: Higher values make the system reward large sample sizes,
    #    increasing confidence for well-sampled zips. Lower values reduce the
    #    impact of sample size.
    #  - Technical: Used in sample_confidence calculation in
    #    _calculate_knn_confidence; higher values increase confidence_level for
    #    neighbor pools with more data points.

    # 🌍 GEOGRAPHIC FACTORS
    PORT_PROXIMITY_BONUS = 5  # Confidence bonus for zip codes near port (< 50 miles).
    #  - Small (<5): Proximity bonus is minor.
    #  - Large (>20): Proximity bonus is major.
    #  - Typical: 8-15.
    #  - Effect: Higher values make the system trust estimates near the port
    #    more, increasing confidence for port-adjacent zips. Lower values reduce
    #    the impact of proximity.
    #  - Technical: Added to geographic_confidence in _calculate_knn_confidence
    #    for zip codes within 50 miles of the port, directly increasing
    #    confidence_level for those estimates.

    URBAN_DENSITY_FACTOR = 1.00  # Rate multiplier for high-density urban areas.
    #  - Small (<1.05): Urban areas barely boosted.
    #  - Large (>1.5): Urban areas strongly boosted.
    #  - Typical: 1.1-1.3.
    #  - Effect: Higher values make the system increase rates for urban areas,
    #    reflecting higher costs. Lower values keep rates more uniform across
    #    regions.
    #  - Technical: Multiplies estimated RPM for zip codes identified as
    #    high-density urban in estimation logic (not shown in this snippet, but
    #    used in rate adjustment formulas).

    RURAL_DISTANCE_PENALTY = 0.05  # Confidence penalty per mile for rural areas (> 200 miles).
    #  - Small (<0.05): Rural penalty is minor.
    #  - Large (>0.3): Rural penalty is major.
    #  - Typical: 0.08-0.15.
    #  - Effect: Higher values make the system penalize confidence for
    #    rural/remote areas more, reflecting increased uncertainty. Lower values
    #    allow more trust in rural estimates.
    #  - Technical: Used to reduce geographic_confidence in
    #    _calculate_knn_confidence for zip codes farther than 200 miles from
    #    the port, decreasing confidence_level for remote locations.

# Initialize global config
CONFIG = EstimatorConfig()

@dataclass
class RateEstimate:
    """Container for rate estimation results"""
    zip_code: str
    estimated_rpm: float
    confidence_level: float
    confidence_category: str
    method_used: str
    nearest_neighbors: List[str]
    distance_to_port: float
    explanation: str
    rate_range: Tuple[float, float]  # (low, high) estimate

class IntelligentRateEstimator:
    """Advanced rate estimation system for unknown zip codes"""
    
    def __init__(self, data_file: str = None, zip_coords_file: str = None):
        """Initialize the rate estimator with data files"""
        # Load data files
        if data_file is None:
            data_file = "../data/port_drayage_dummy_data.csv"
        if zip_coords_file is None:
            zip_coords_file = "../data/us_zip_coordinates.csv"
            
        print("Loading data files...")
        self.drayage_data = pd.read_csv(data_file)
        self.zip_coords = pd.read_csv(zip_coords_file, dtype={'ZIP': str})
        
        # Get LA Port coordinates from CSV (zip code 90802)
        port_info = self.zip_coords[self.zip_coords['ZIP'] == '90802']
        if not port_info.empty:
            self.port_location = (port_info.iloc[0]['LAT'], port_info.iloc[0]['LNG'])
        else:
            raise ValueError("Port zip code 90802 not found in coordinate database")
        
        # Prepare data
        self._prepare_data()
        self._build_spatial_index()
        
        print(f"✓ Loaded {len(self.drayage_data)} drayage records")
        print(f"✓ Loaded {len(self.zip_coords)} zip code coordinates")
        print(f"✓ Found {len(self.known_zips)} unique zip codes with rate data")
        
    def _prepare_data(self):
        """Prepare data for estimation with time weighting"""
        # Get current date for time calculations
        current_date = datetime.now()
        
        # Convert date strings to datetime objects
        self.drayage_data['date'] = pd.to_datetime(self.drayage_data['date'])
        
        # Calculate days since each transaction
        self.drayage_data['days_since'] = (current_date - self.drayage_data['date']).dt.days
        
        # Calculate time weights using exponential decay
        self.drayage_data['time_weight'] = self._calculate_time_weight(self.drayage_data['days_since'])
        
        # Combine origin and destination data for comprehensive view
        origins = self.drayage_data[['origin_zip', 'origin_lat', 'origin_lng', 'RPM', 'date', 'days_since', 'time_weight']].copy()
        origins.columns = ['zip', 'lat', 'lng', 'rpm', 'date', 'days_since', 'time_weight']
        origins['direction'] = 'origin'
        
        destinations = self.drayage_data[['destination_zip', 'destination_lat', 'destination_lng', 'RPM', 'date', 'days_since', 'time_weight']].copy()
        destinations.columns = ['zip', 'lat', 'lng', 'rpm', 'date', 'days_since', 'time_weight']
        destinations['direction'] = 'destination'
        
        # Combine all rate data
        self.all_rates = pd.concat([origins, destinations], ignore_index=True)
        self.all_rates = self.all_rates.dropna()
        
        # Calculate distance from port for each zip
        self.all_rates['distance_to_port'] = self.all_rates.apply(
            lambda row: geodesic(self.port_location, (row['lat'], row['lng'])).miles, axis=1
        )
        
        # Get unique zip codes with data
        self.known_zips = set(self.all_rates['zip'].unique())
        
        # Create time-weighted zip code summary statistics
        self.zip_stats = self._create_weighted_zip_stats()
        
    def _calculate_time_weight(self, days_since: pd.Series) -> pd.Series:
        """Calculate time-based weights for data points"""
        weights = np.ones(len(days_since))
        
        # Apply recent boost for very recent data
        recent_mask = days_since <= CONFIG.RECENT_BOOST_DAYS
        weights[recent_mask] *= CONFIG.RECENT_MULTIPLIER
        
        # Apply exponential decay for older data
        decay_mask = days_since > CONFIG.RECENT_BOOST_DAYS
        decay_days = days_since[decay_mask] - CONFIG.RECENT_BOOST_DAYS
        weights[decay_mask] *= (CONFIG.TIME_DECAY_RATE ** decay_days)
        
        # Cap minimum weight to prevent complete dismissal of old data
        weights = np.maximum(weights, 0.05)
        
        return pd.Series(weights, index=days_since.index)
    
    def _create_weighted_zip_stats(self):
        """Create zip code statistics with time weighting"""
        zip_groups = self.all_rates.groupby('zip')
        
        stats_list = []
        for zip_code, group in zip_groups:
            # Calculate time-weighted statistics
            total_weight = group['time_weight'].sum()
            
            if total_weight > 0:
                weighted_rpm = (group['rpm'] * group['time_weight']).sum() / total_weight
                
                # Calculate weighted variance
                weighted_var = ((group['rpm'] - weighted_rpm) ** 2 * group['time_weight']).sum() / total_weight
                weighted_std = np.sqrt(weighted_var)
                
                # Get most recent transaction info
                most_recent_idx = group['days_since'].idxmin()
                most_recent = group.loc[most_recent_idx]
                
                stats_list.append({
                    'zip': zip_code,
                    'rpm_mean': weighted_rpm,
                    'rpm_std': weighted_std,
                    'rpm_count': len(group),
                    'rpm_min': group['rpm'].min(),
                    'rpm_max': group['rpm'].max(),
                    'lat_first': most_recent['lat'],
                    'lng_first': most_recent['lng'],
                    'distance_to_port_first': most_recent['distance_to_port'],
                    'most_recent_days': most_recent['days_since'],
                    'total_time_weight': total_weight,
                    'avg_time_weight': total_weight / len(group)
                })
        
        return pd.DataFrame(stats_list)
        
    def _build_spatial_index(self):
        """Build spatial index for efficient neighbor searching"""
        # Create distance-based model
        self.distance_model = self._fit_distance_model()
        
    def _fit_distance_model(self):
        """Fit a distance-based rate model"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Prepare features
        features = ['distance_to_port', 'lat', 'lng']
        X = self.all_rates[features].copy()
        y = self.all_rates['rpm'].copy()
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✓ Distance model trained - MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return model
        
    def estimate_rate(self, zip_code: str, verbose: bool = True) -> RateEstimate:
        """
        Estimate rate for a zip code with no historical data
        
        Args:
            zip_code: Target zip code (as string)
            verbose: Print detailed explanation
            
        Returns:
            RateEstimate object with prediction and confidence
        """
        # Convert zip code to integer for lookup, then back to string for coordinate lookup
        try:
            zip_int = int(zip_code)
            zip_str = f"{zip_int:05d}"  # Format as 5-digit string with leading zeros
        except ValueError:
            raise ValueError(f"Invalid zip code format: {zip_code}")
            
        # Check if we already have data for this zip
        if zip_int in self.known_zips:
            return self._get_known_rate(zip_int)
            
        # Get coordinates for target zip
        zip_info = self.zip_coords[self.zip_coords['ZIP'] == zip_str]
        if zip_info.empty:
            raise ValueError(f"Zip code {zip_code} not found in coordinate database")
            
        target_lat = zip_info.iloc[0]['LAT']
        target_lng = zip_info.iloc[0]['LNG']
        target_location = (target_lat, target_lng)
        
        # Calculate distance to port
        distance_to_port = geodesic(self.port_location, target_location).miles
        
        if verbose:
            print(f"\n🎯 Estimating rate for zip code: {zip_code}")
            print(f"📍 Location: ({target_lat:.4f}, {target_lng:.4f})")
            print(f"🚢 Distance to LA Port: {distance_to_port:.1f} miles")
        
        # Method 1: K-Nearest Neighbors
        knn_result = self._knn_estimation(target_location, k=5)
        
        # Method 2: Distance Model
        distance_result = self._distance_model_estimation(target_lat, target_lng, distance_to_port)
        
        # Combine methods and calculate confidence
        final_estimate = self._combine_estimates(
            knn_result, distance_result, 
            target_location, distance_to_port, zip_code, verbose
        )
        
        if verbose:
            print(f"\n📊 Final Estimate:")
            print(f"💰 Estimated RPM: ${final_estimate.estimated_rpm:.2f}")
            print(f"🎯 Confidence: {final_estimate.confidence_level:.1f}% ({final_estimate.confidence_category})")
            print(f"📈 Rate Range: ${final_estimate.rate_range[0]:.2f} - ${final_estimate.rate_range[1]:.2f}")
            print(f"🔧 Method: {final_estimate.method_used}")
            print(f"\n💡 {final_estimate.explanation}")
        
        return final_estimate
        
    def _get_known_rate(self, zip_code: int) -> RateEstimate:
        """Get rate estimate for zip with existing data"""
        zip_data = self.zip_stats[self.zip_stats['zip'] == zip_code].iloc[0]
        
        avg_rpm = zip_data['rpm_mean']
        std_rpm = zip_data['rpm_std'] if not pd.isna(zip_data['rpm_std']) else 0
        count = zip_data['rpm_count']
        
        # High confidence for known zips
        confidence = min(95, 60 + (count * 5))  # More data = higher confidence
        
        rate_range = (
            max(0, avg_rpm - std_rpm),
            avg_rpm + std_rpm
        )
        
        return RateEstimate(
            zip_code=str(zip_code),
            estimated_rpm=avg_rpm,
            confidence_level=confidence,
            confidence_category="Very High",
            method_used="Historical Data",
            nearest_neighbors=[str(zip_code)],
            distance_to_port=zip_data['distance_to_port_first'],
            explanation=f"Based on {int(count)} historical transactions. Average RPM: ${avg_rpm:.2f}",
            rate_range=rate_range
        )
    
    def _knn_estimation(self, target_location: Tuple[float, float], k: int = None) -> Dict:
        """Enhanced K-Nearest Neighbors estimation with time and port distance similarity weighting"""
        if k is None:
            k = CONFIG.KNN_K_NEIGHBORS
        
        # Calculate target location's distance to port
        target_port_distance = geodesic(self.port_location, target_location).miles
            
        # Calculate distances to all known zips
        distances = []
        for _, row in self.zip_stats.iterrows():
            zip_location = (row['lat_first'], row['lng_first'])
            geographic_dist = geodesic(target_location, zip_location).miles
            
            # Skip neighbors that are too far away geographically
            if geographic_dist > CONFIG.MAX_NEIGHBOR_DISTANCE:
                continue
            
            # Calculate port distance similarity
            neighbor_port_distance = row['distance_to_port_first']
            port_distance_diff = abs(target_port_distance - neighbor_port_distance)
                
            distances.append({
                'zip': row['zip'],
                'geographic_distance': geographic_dist,
                'port_distance': neighbor_port_distance,
                'port_distance_diff': port_distance_diff,
                'rpm': row['rpm_mean'],
                'count': row['rpm_count'],
                'std': row['rpm_std'] if not pd.isna(row['rpm_std']) else 0,
                'recency_days': row['most_recent_days'],
                'time_weight_avg': row['avg_time_weight'],
                'total_weight': row['total_time_weight']
            })
        
        if not distances:
            return {'rpm': None, 'confidence': 0, 'neighbors': [], 'reason': 'No neighbors within range'}
        
        # Sort by combined distance metric (geographic + port distance similarity)
        for neighbor in distances:
            # Calculate combined distance score for sorting
            geo_score = neighbor['geographic_distance']
            port_score = neighbor['port_distance_diff'] * 0.5  # Port distance difference weighted
            neighbor['combined_distance_score'] = geo_score + port_score
        
        distances.sort(key=lambda x: x['combined_distance_score'])
        nearest = distances[:k]
        
        # Calculate enhanced weights combining geographic distance, port distance similarity, and time
        total_weight = 0
        weighted_rpm = 0
        
        for neighbor in nearest:
            # Geographic distance weight (inverse distance with decay)
            geographic_weight = 1 / (neighbor['geographic_distance'] + 1) ** CONFIG.DISTANCE_DECAY_FACTOR
            geographic_weight = max(geographic_weight, CONFIG.MIN_DISTANCE_WEIGHT)
            
            # Port distance similarity weight (KEY ENHANCEMENT)
            port_dist_diff = neighbor['port_distance_diff']
            if port_dist_diff > CONFIG.MAX_PORT_DISTANCE_DIFFERENCE:
                port_distance_weight = CONFIG.MIN_PORT_DISTANCE_WEIGHT
            else:
                port_distance_weight = 1 / (1 + port_dist_diff) ** CONFIG.PORT_DISTANCE_DECAY_FACTOR
                port_distance_weight = max(port_distance_weight, CONFIG.MIN_PORT_DISTANCE_WEIGHT)
            
            # Apply port distance similarity weighting factor
            port_distance_weight *= CONFIG.PORT_DISTANCE_SIMILARITY_WEIGHT
            
            # Time-based weight (recent data gets higher weight)
            recency_weight = self._calculate_recency_weight(neighbor['recency_days'])
            
            # Sample size weight (more data points = higher confidence)
            sample_weight = min(1.0 + (neighbor['count'] - 1) * 0.1, 2.0)
            
            # Combined weight (now includes port distance similarity)
            combined_weight = (geographic_weight * port_distance_weight * 
                             recency_weight * sample_weight * neighbor['time_weight_avg'])
            
            weighted_rpm += neighbor['rpm'] * combined_weight
            total_weight += combined_weight
            
            # Store weights for debugging
            neighbor['geographic_weight'] = geographic_weight
            neighbor['port_distance_weight'] = port_distance_weight
            neighbor['combined_weight'] = combined_weight
        
        if total_weight > 0:
            weighted_rpm /= total_weight
        
        # Enhanced confidence calculation including port distance similarity
        confidence = self._calculate_knn_confidence(nearest, target_location, target_port_distance)
        # Calculate percentage variance for return info
        rpm_values = [n['rpm'] for n in nearest]
        rpm_mean = np.mean(rpm_values)
        rpm_variance = np.var(rpm_values)
        rpm_percentage_variance = (np.sqrt(rpm_variance) / rpm_mean) * 100 if rpm_mean > 0 else 100
        
        return {
            'rpm': weighted_rpm,
            'confidence': confidence,
            'neighbors': [str(n['zip']) for n in nearest],
            'avg_distance': np.mean([n['geographic_distance'] for n in nearest]),
            'avg_port_distance_diff': np.mean([n['port_distance_diff'] for n in nearest]),
            'rpm_variance': rpm_variance,
            'rpm_percentage_variance': rpm_percentage_variance,
            'avg_recency': np.mean([n['recency_days'] for n in nearest]),
            'total_sample_size': sum([n['count'] for n in nearest]),
            'target_port_distance': target_port_distance,
            'neighbor_details': nearest  # Include detailed neighbor info for debugging
        }
    
    def _calculate_recency_weight(self, days_since: float) -> float:
        """Calculate recency weight for a data point"""
        if days_since <= CONFIG.RECENT_BOOST_DAYS:
            return CONFIG.RECENT_MULTIPLIER
        elif days_since <= CONFIG.TIME_DECAY_DAYS:
            decay_days = days_since - CONFIG.RECENT_BOOST_DAYS
            return CONFIG.TIME_DECAY_RATE ** decay_days
        else:
            # Very old data gets minimal weight
            return 0.1
    
    def _calculate_knn_confidence(self, neighbors: List[Dict], target_location: Tuple[float, float], 
                                 target_port_distance: float) -> float:
        """Calculate confidence for KNN estimation using enhanced metrics including port distance similarity"""
        if not neighbors:
            return 0
        
        # Base confidence from neighbor proximity (geographic)
        avg_geographic_distance = np.mean([n['geographic_distance'] for n in neighbors])
        distance_confidence = max(0, 100 - avg_geographic_distance * 2)
        
        # Port distance similarity confidence (NEW - KEY ENHANCEMENT)
        avg_port_distance_diff = np.mean([n['port_distance_diff'] for n in neighbors])
        if avg_port_distance_diff < 10:  # Very similar port distances
            port_similarity_confidence = 95
        elif avg_port_distance_diff < 25:  # Moderately similar port distances
            port_similarity_confidence = 80
        elif avg_port_distance_diff < 50:  # Somewhat similar port distances
            port_similarity_confidence = 60
        else:  # Very different port distances
            port_similarity_confidence = 30
        
        # Recency confidence (more recent data = higher confidence)
        avg_recency = np.mean([n['recency_days'] for n in neighbors])
        if avg_recency <= CONFIG.RECENT_BOOST_DAYS:
            recency_confidence = 95
        elif avg_recency <= CONFIG.TIME_DECAY_DAYS:
            recency_confidence = max(50, 95 - (avg_recency - CONFIG.RECENT_BOOST_DAYS) * 0.3)
        else:
            recency_confidence = 30
        
        # Consistency confidence (low percentage variance = higher confidence) - Distance-sensitive
        rpm_values = [n['rpm'] for n in neighbors]
        rpm_mean = np.mean(rpm_values)
        rpm_variance = np.var(rpm_values)
        
        # Calculate percentage variance (coefficient of variation * 100)
        if rpm_mean > 0:
            rpm_percentage_variance = (np.sqrt(rpm_variance) / rpm_mean) * 100
        else:
            rpm_percentage_variance = 100  # High penalty for zero/negative mean
        
        # Calculate dynamic percentage variance threshold based on distance from port
        dynamic_percentage_threshold = (CONFIG.VARIANCE_THRESHOLD_BASE_PERCENT + 
                                      target_port_distance * CONFIG.VARIANCE_DISTANCE_SCALING_PERCENT)
        
        if rpm_percentage_variance < dynamic_percentage_threshold:
            consistency_confidence = 90 + CONFIG.CONSISTENCY_BONUS
        else:
            # Penalize based on how much percentage variance exceeds the dynamic threshold
            excess_percentage_variance = rpm_percentage_variance - dynamic_percentage_threshold
            consistency_confidence = max(30, 90 - excess_percentage_variance * CONFIG.HIGH_VARIANCE_PENALTY / 10)
        
        # Sample size confidence
        total_samples = sum([n['count'] for n in neighbors])
        sample_confidence = min(90, 50 + total_samples * CONFIG.SAMPLE_SIZE_BOOST_FACTOR)
        
        # Geographic confidence (closer to port = higher confidence for this domain)
        if target_port_distance < 50:
            geographic_confidence = 90 + CONFIG.PORT_PROXIMITY_BONUS
        elif target_port_distance < 200:
            geographic_confidence = 80
        else:
            geographic_confidence = max(40, 80 - (target_port_distance - 200) * CONFIG.RURAL_DISTANCE_PENALTY)
        
        # Weighted combination of confidence factors (updated weights to include port similarity)
        weights = [0.25, 0.25, 0.2, 0.15, 0.1, 0.05]  # distance, port_similarity, recency, consistency, sample_size, geographic
        confidences = [distance_confidence, port_similarity_confidence, recency_confidence, 
                      consistency_confidence, sample_confidence, geographic_confidence]
        
        final_confidence = sum(w * c for w, c in zip(weights, confidences))
        
        return min(100, max(0, final_confidence))
    
    def _distance_model_estimation(self, lat: float, lng: float, distance: float) -> Dict:
        """Enhanced distance-based model estimation with time weighting"""
        try:
            # Include time weights in model features
            features = ['distance_to_port', 'lat', 'lng', 'time_weight']
            X = self.all_rates[features].copy()
            y = self.all_rates['rpm'].copy()
            
            # Apply sample weights based on time weighting
            sample_weights = self.all_rates['time_weight'].values
            
            # Retrain model with time weights
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y, sample_weight=sample_weights)
            
            # Make prediction with average time weight for new data
            avg_time_weight = self.all_rates['time_weight'].mean()
            features_pred = np.array([[distance, lat, lng, avg_time_weight]])
            predicted_rpm = model.predict(features_pred)[0]
            
            # Enhanced confidence calculation
            training_distances = self.all_rates['distance_to_port']
            min_dist, max_dist = training_distances.min(), training_distances.max()
            
            base_confidence = CONFIG.ML_MODEL_CONFIDENCE_BASE
            
            if min_dist <= distance <= max_dist:
                # Within training range - high confidence
                confidence = base_confidence
                
                # Bonus for areas with more recent data
                nearby_data = self.all_rates[
                    (abs(self.all_rates['distance_to_port'] - distance) < 20) &
                    (self.all_rates['days_since'] <= CONFIG.RECENT_BOOST_DAYS)
                ]
                if len(nearby_data) > 0:
                    confidence += 15
                    
            else:
                # Extrapolation - reduce confidence
                if distance < min_dist:
                    extrapolation_distance = min_dist - distance
                else:
                    extrapolation_distance = distance - max_dist
                
                confidence = max(20, base_confidence - extrapolation_distance * CONFIG.ML_EXTRAPOLATION_PENALTY)
            
            return {
                'rpm': predicted_rpm,
                'confidence': confidence,
                'in_range': min_dist <= distance <= max_dist,
                'extrapolation_distance': extrapolation_distance if distance < min_dist or distance > max_dist else 0
            }
            
        except Exception as e:
            return {'rpm': None, 'confidence': 0, 'error': str(e)}
    
    def _combine_estimates(self, knn_result: Dict, distance_result: Dict, 
                          target_location: Tuple, distance_to_port: float, 
                          zip_code: str, verbose: bool) -> RateEstimate:
        """Combine multiple estimation methods"""
        
        estimates = []
        confidences = []
        primary_method = "Unknown"  # Initialize with default value
        neighbors = []  # Initialize with default value
        
        # KNN estimate
        if knn_result['rpm'] is not None:
            estimates.append(knn_result['rpm'])
            confidences.append(knn_result['confidence'])
            primary_method = f"K-Nearest Neighbors (avg distance: {knn_result['avg_distance']:.1f} miles)"
            neighbors = knn_result['neighbors']
        
        # Distance model estimate  
        if distance_result['rpm'] is not None:
            estimates.append(distance_result['rpm'])
            confidences.append(distance_result['confidence'])
            if primary_method == "Unknown" or distance_result['confidence'] > confidences[0]:
                primary_method = "Distance-based ML Model"
        
        if not estimates:
            # Ultimate fallback
            final_rpm = 8.0  # Conservative estimate
            final_confidence = 10
            primary_method = "System Default"
            rate_range = (6.0, 10.0)
            explanation = "Very limited data available. Using conservative system default."
        else:
            # Weight estimates by confidence
            total_weight = sum(confidences)
            if total_weight > 0:
                final_rpm = sum(est * conf for est, conf in zip(estimates, confidences)) / total_weight
                final_confidence = max(confidences)
            else:
                final_rpm = np.mean(estimates)
                final_confidence = np.mean(confidences) if confidences else 20
            
            # Calculate rate range
            rpm_std = np.std(estimates) if len(estimates) > 1 else final_rpm * 0.15
            rate_range = (
                max(0, final_rpm - rpm_std),
                final_rpm + rpm_std
            )
            
            # Generate explanation
            if final_confidence >= 80:
                confidence_category = "Very High"
                explanation = f"High confidence estimate based on nearby zip codes with consistent rates."
            elif final_confidence >= 60:
                confidence_category = "High"  
                explanation = f"Good estimate based on nearby data and distance modeling."
            elif final_confidence >= 40:
                confidence_category = "Medium"
                explanation = f"Moderate confidence. Limited nearby data, relying on distance patterns."
            else:
                confidence_category = "Low"
                explanation = f"Low confidence. Sparse data in area, estimate based on general patterns."
        
        if verbose:
            print(f"\n🔍 Estimation Methods Used:")
            if knn_result['rpm'] is not None:
                print(f"  • KNN: ${knn_result['rpm']:.2f} (confidence: {knn_result['confidence']:.1f}%)")
                print(f"    - Avg geographic distance: {knn_result['avg_distance']:.1f} miles")
                print(f"    - Avg port distance difference: {knn_result['avg_port_distance_diff']:.1f} miles")
                print(f"    - Target port distance: {knn_result['target_port_distance']:.1f} miles")
            if distance_result['rpm'] is not None:
                print(f"  • Distance Model: ${distance_result['rpm']:.2f} (confidence: {distance_result['confidence']:.1f}%)")
        
        return RateEstimate(
            zip_code=zip_code,
            estimated_rpm=round(final_rpm, 2),
            confidence_level=round(final_confidence, 1),
            confidence_category=confidence_category,
            method_used=primary_method,
            nearest_neighbors=neighbors[:3],  # Top 3 neighbors
            distance_to_port=round(distance_to_port, 1),
            explanation=explanation,
            rate_range=(round(rate_range[0], 2), round(rate_range[1], 2))
        )

def demo_rate_estimator():
    """Demonstrate the rate estimator with example zip codes"""
    print("🚛 DrayVis Intelligent Rate Estimator Demo")
    print("=" * 50)
    
    # Initialize estimator
    estimator = IntelligentRateEstimator()
    
    # Test with various zip codes
    test_zips = [
        "90210",  # Beverly Hills - should have good nearby data
        "93536",  # Antelope Valley - moderate distance
        "96001",  # Redding - far from port
        "85001",  # Phoenix - very far, different region
        "10001"   # NYC - extremely far
    ]
    
    print("\n🧪 Testing Rate Estimation for Various Zip Codes:")
    print("=" * 60)
    
    results = []
    
    for zip_code in test_zips:
        try:
            print(f"\n{'='*60}")
            estimate = estimator.estimate_rate(zip_code, verbose=True)
            results.append(estimate)
            
        except Exception as e:
            print(f"❌ Error estimating rate for {zip_code}: {e}")
    
    # Summary table
    print(f"\n{'='*80}")
    print("📋 SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Zip Code':<8} {'Est. RPM':<10} {'Confidence':<12} {'Distance':<10} {'Category':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result.zip_code:<8} ${result.estimated_rpm:<9.2f} {result.confidence_level:<11.1f}% "
              f"{result.distance_to_port:<9.1f}mi {result.confidence_category:<15}")
    
    print("\n✅ Rate estimation demo completed!")

if __name__ == "__main__":
    demo_rate_estimator()
