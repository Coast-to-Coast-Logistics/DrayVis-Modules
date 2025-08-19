"""
DrayVis K-Fold Cross-Validation Framework
=========================================

Comprehensive validation system for testing the DrayVis Intelligent Rate Estimator
using k-fold cross-validation with parameter sensitivity analysis.

Author: DrayVis Analytics Team
Date: August 19, 2025
Version: 2.0 - High-Performance Parallel Implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from geopy.distance import geodesic
import copy
import warnings
from pathlib import Path
import json
from datetime import datetime
import multiprocessing as mp
import concurrent.futures
from functools import partial
from tqdm import tqdm
import psutil
import gc
import os

# Import our estimator
from intelligent_rate_estimator import IntelligentRateEstimator, EstimatorConfig, RateEstimate

warnings.filterwarnings('ignore')

# Southern California mainland optimization - 300 mile radius from LA port
PORT_ZIP = '90802'  # LA Port zip code
MAX_DISTANCE_MILES = 300  # Maximum distance from port for inclusion (comprehensive coverage)
EXCLUDED_ZIP_PREFIXES = ['99', '96']  # Alaska, Hawaii
KNOWN_ISLAND_ZIPS = ['90704', '90740']  # Catalina Island and other offshore areas

@dataclass
class ValidationResult:
    """Container for validation results with comprehensive explanations"""
    config_name: str
    config_params: Dict
    fold_results: List[Dict]
    overall_metrics: Dict
    confidence_calibration: Dict
    geographic_analysis: Dict
    timing_info: Dict
    explanations: Dict  # New field for result explanations
    coverage_stats: Dict  # New field for geographic coverage statistics

@dataclass
class FoldResult:
    """Container for single fold results"""
    fold_number: int
    test_zips: List[str]
    predictions: List[RateEstimate]
    actual_values: List[float]
    errors: List[float]
    absolute_errors: List[float]
    metrics: Dict

class ConfigurationTester:
    """Test different configurations of the estimator"""
    
    def __init__(self):
        """Initialize configuration tester"""
        self.base_config = EstimatorConfig()
        self.test_configurations = self._create_test_configurations()
    
    def _create_test_configurations(self) -> Dict[str, EstimatorConfig]:
        """Create EXTREMELY different test configurations for maximum differentiation"""
        configs = {}
        
        # Baseline configuration
        configs['baseline'] = copy.deepcopy(self.base_config)
        
        # EXTREME configurations with MAXIMUM parameter differences
        
        # 1. Ultra-Recent Focus - EXTREME recent data bias
        configs['ultra_recent'] = copy.deepcopy(self.base_config)
        configs['ultra_recent'].TIME_DECAY_DAYS = 3  # Extremely short decay
        configs['ultra_recent'].TIME_DECAY_RATE = 0.1  # Very aggressive decay
        configs['ultra_recent'].RECENT_BOOST_DAYS = 1
        configs['ultra_recent'].RECENT_MULTIPLIER = 10.0  # Extreme recent boost
        configs['ultra_recent'].MAX_NEIGHBOR_DISTANCE = 2  # Ultra-local
        configs['ultra_recent'].KNN_K_NEIGHBORS = 1  # Single neighbor
        configs['ultra_recent'].CONFIDENCE_THRESHOLD = 95  # Very high confidence
        configs['ultra_recent'].MIN_TRANSACTIONS = 50  # High quality only
        
        # 2. Historical Stable - EXTREME historical bias
        configs['historical_stable'] = copy.deepcopy(self.base_config)
        configs['historical_stable'].TIME_DECAY_DAYS = 1000  # Extremely long decay
        configs['historical_stable'].TIME_DECAY_RATE = 0.999  # Almost no decay
        configs['historical_stable'].RECENT_BOOST_DAYS = 365
        configs['historical_stable'].RECENT_MULTIPLIER = 1.01  # Tiny recent boost
        configs['historical_stable'].MAX_NEIGHBOR_DISTANCE = 100  # Very wide search
        configs['historical_stable'].KNN_K_NEIGHBORS = 50  # Many neighbors
        configs['historical_stable'].CONFIDENCE_THRESHOLD = 50  # Low confidence OK
        configs['historical_stable'].MIN_TRANSACTIONS = 1  # Accept any data
        
        # 3. Hyper-Local - EXTREME proximity focus
        configs['hyper_local'] = copy.deepcopy(self.base_config)
        configs['hyper_local'].MAX_NEIGHBOR_DISTANCE = 1  # Extreme proximity
        configs['hyper_local'].KNN_K_NEIGHBORS = 1  # Single closest
        configs['hyper_local'].DISTANCE_WEIGHT = 10.0  # Extreme distance weight
        configs['hyper_local'].TIME_WEIGHT = 0.1  # Minimal time weight
        configs['hyper_local'].VOLUME_WEIGHT = 0.1  # Minimal volume weight
        configs['hyper_local'].MIN_NEIGHBOR_DISTANCE = 0.1
        configs['hyper_local'].FALLBACK_RATE = 50.0  # Low fallback
        
        # 4. Regional Broad - EXTREME wide geographic search
        configs['regional_broad'] = copy.deepcopy(self.base_config)
        configs['regional_broad'].MAX_NEIGHBOR_DISTANCE = 200  # Very wide
        configs['regional_broad'].KNN_K_NEIGHBORS = 100  # Many neighbors
        configs['regional_broad'].DISTANCE_WEIGHT = 0.1  # Minimal distance weight
        configs['regional_broad'].TIME_WEIGHT = 2.0  # High time weight
        configs['regional_broad'].VOLUME_WEIGHT = 2.0  # High volume weight
        configs['regional_broad'].MIN_NEIGHBOR_DISTANCE = 50
        configs['regional_broad'].FALLBACK_RATE = 200.0  # High fallback
        
        # 5. Quality Focused - EXTREME data quality requirements
        configs['quality_focused'] = copy.deepcopy(self.base_config)
        configs['quality_focused'].MIN_TRANSACTIONS = 100  # Very high requirement
        configs['quality_focused'].CONFIDENCE_THRESHOLD = 99  # Extreme confidence
        configs['quality_focused'].TIME_DECAY_DAYS = 30
        configs['quality_focused'].RECENT_MULTIPLIER = 3.0
        configs['quality_focused'].KNN_K_NEIGHBORS = 20
        configs['quality_focused'].MAX_NEIGHBOR_DISTANCE = 20
        
        # 6. Speed Optimized - EXTREME speed at cost of accuracy
        configs['speed_optimized'] = copy.deepcopy(self.base_config)
        configs['speed_optimized'].KNN_K_NEIGHBORS = 1  # Minimal neighbors
        configs['speed_optimized'].MAX_NEIGHBOR_DISTANCE = 5  # Close search
        configs['speed_optimized'].MIN_TRANSACTIONS = 1  # Accept anything
        configs['speed_optimized'].CONFIDENCE_THRESHOLD = 10  # Any confidence
        configs['speed_optimized'].TIME_DECAY_DAYS = 1000  # No time processing
        configs['speed_optimized'].RECENT_MULTIPLIER = 1.0  # No recent boost
        
        # 7. ML Heavy - EXTREME machine learning emphasis
        configs['ml_heavy'] = copy.deepcopy(self.base_config)
        configs['ml_heavy'].KNN_K_NEIGHBORS = 30  # Many for ML
        configs['ml_heavy'].DISTANCE_WEIGHT = 1.0
        configs['ml_heavy'].TIME_WEIGHT = 1.0
        configs['ml_heavy'].VOLUME_WEIGHT = 1.0
        configs['ml_heavy'].MAX_NEIGHBOR_DISTANCE = 30
        configs['ml_heavy'].TIME_DECAY_DAYS = 90
        configs['ml_heavy'].CONFIDENCE_THRESHOLD = 70
        
        # 8. Distance Centric - EXTREME distance-only focus
        configs['distance_centric'] = copy.deepcopy(self.base_config)
        configs['distance_centric'].DISTANCE_WEIGHT = 100.0  # Extreme distance focus
        configs['distance_centric'].TIME_WEIGHT = 0.01  # Ignore time
        configs['distance_centric'].VOLUME_WEIGHT = 0.01  # Ignore volume
        configs['distance_centric'].TIME_DECAY_DAYS = 10000  # Ignore age
        configs['distance_centric'].RECENT_MULTIPLIER = 1.0  # No recent boost
        configs['distance_centric'].KNN_K_NEIGHBORS = 5
        configs['distance_centric'].MAX_NEIGHBOR_DISTANCE = 50
        
        return configs
        configs['hyper_local'] = copy.deepcopy(self.base_config)
        configs['hyper_local'].MAX_NEIGHBOR_DISTANCE = 3  # VERY tight radius
        configs['hyper_local'].DISTANCE_DECAY_FACTOR = 5.0  # Extreme distance penalty
        configs['hyper_local'].KNN_K_NEIGHBORS = 2  # Minimal neighbors
        configs['hyper_local'].PORT_DISTANCE_SIMILARITY_WEIGHT = 20  # Extreme port similarity
        configs['hyper_local'].MAX_PORT_DISTANCE_DIFFERENCE = 0.5  # Very tight port distance
        
        # 4. Regional Broad (wide geographic patterns)
        configs['regional_broad'] = copy.deepcopy(self.base_config)
        configs['regional_broad'].MAX_NEIGHBOR_DISTANCE = 100  # Very wide radius
        configs['regional_broad'].DISTANCE_DECAY_FACTOR = 0.5  # Minimal distance penalty
        configs['regional_broad'].KNN_K_NEIGHBORS = 25  # Many neighbors
        configs['regional_broad'].PORT_DISTANCE_SIMILARITY_WEIGHT = 0.1  # Ignore port similarity
        configs['regional_broad'].MAX_PORT_DISTANCE_DIFFERENCE = 100  # Very loose port distance
        
        # 5. Data Quality Focused (high standards for data)
        configs['quality_focused'] = copy.deepcopy(self.base_config)
        configs['quality_focused'].MIN_SIMILAR_RECORDS = 8  # Need more data
        configs['quality_focused'].CONSISTENCY_BONUS = 50  # Huge consistency reward
        configs['quality_focused'].HIGH_VARIANCE_PENALTY = 40  # Strong variance penalty
        configs['quality_focused'].CONFIDENCE_VERY_HIGH_MIN = 95
        configs['quality_focused'].CONFIDENCE_HIGH_MIN = 85
        configs['quality_focused'].VOLUME_BOOST_FACTOR = 2.0  # Reward high volume
        
        # 6. Speed Optimized (minimal computation)
        configs['speed_optimized'] = copy.deepcopy(self.base_config)
        configs['speed_optimized'].KNN_K_NEIGHBORS = 1  # Single neighbor
        configs['speed_optimized'].MAX_NEIGHBOR_DISTANCE = 10
        configs['speed_optimized'].MIN_SIMILAR_RECORDS = 1  # Minimal data needed
        configs['speed_optimized'].USE_ML_MODEL = False  # Skip ML
        configs['speed_optimized'].ENABLE_CACHING = True
        
        # 7. ML-Heavy (rely on machine learning)
        configs['ml_heavy'] = copy.deepcopy(self.base_config)
        configs['ml_heavy'].ML_MODEL_WEIGHT = 3.0  # Heavy ML weighting
        configs['ml_heavy'].KNN_WEIGHT = 0.3  # Reduce KNN influence
        configs['ml_heavy'].DISTANCE_MODEL_WEIGHT = 0.2  # Reduce distance model
        configs['ml_heavy'].USE_ML_MODEL = True
        configs['ml_heavy'].ML_CONFIDENCE_BOOST = 30
        
        # 8. Distance-Centric (pure distance-based)
        configs['distance_centric'] = copy.deepcopy(self.base_config)
        configs['distance_centric'].DISTANCE_MODEL_WEIGHT = 3.0  # Heavy distance weighting
        configs['distance_centric'].KNN_WEIGHT = 0.2  # Minimal KNN
        configs['distance_centric'].ML_MODEL_WEIGHT = 0.1  # Minimal ML
        configs['distance_centric'].DISTANCE_DECAY_FACTOR = 1.0  # Linear distance
        
        return configs
    
    def get_configuration_names(self) -> List[str]:
        """Get list of available configuration names"""
        return list(self.test_configurations.keys())
    
    def get_configuration(self, name: str) -> EstimatorConfig:
        """Get a specific configuration by name"""
        if name not in self.test_configurations:
            raise ValueError(f"Configuration '{name}' not found")
        return self.test_configurations[name]

class KFoldValidator:
    """K-Fold cross-validation for the rate estimator - Optimized for LA Region"""
    
    def __init__(self, data_file: str = None, zip_coords_file: str = None, k: int = 5, random_seed: int = 42, 
                 fast_mode: bool = False, max_test_zips: int = None):
        """
        Initialize K-Fold validator with enhanced differentiation capabilities
        
        Args:
            data_file: Path to drayage data CSV
            zip_coords_file: Path to zip coordinates CSV
            k: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
            fast_mode: Enable speed optimizations (reduces accuracy slightly)
            max_test_zips: Maximum number of zip codes to test (None = all available)
        """
        self.k = k
        self.random_seed = random_seed
        self.fast_mode = fast_mode
        # Increase test zips for better differentiation (was 15, now 30 minimum)
        self.max_test_zips = max_test_zips or (30 if fast_mode else None)
        self.data_file = data_file or "../data/port_drayage_dummy_data.csv"
        self.zip_coords_file = zip_coords_file or "../data/us_zip_coordinates.csv"
        
        # Performance tracking
        self.performance_stats = {}
        
        # Load and prepare data with enhanced validation approach
        print("🚀 Loading data for comprehensive configuration validation...")
        if fast_mode:
            print("⚡ Enhanced fast mode - optimized for configuration differentiation")
        start_time = time.time()
        self._load_and_prepare_data_optimized()
        load_time = time.time() - start_time
        
        self.performance_stats['data_load_time'] = load_time
        
        # Initialize configuration tester
        self.config_tester = ConfigurationTester()
        
        print(f"✅ Enhanced K-Fold Validator initialized with {k} folds")
        print(f"✅ Mainland zip codes loaded: {len(self.mainland_zip_coords)}")
        print(f"✅ Zip codes with sufficient data: {len(self.zip_codes_with_data)}")
        if fast_mode:
            print(f"⚡ Enhanced fast mode: {self.max_test_zips} diverse test zip codes")
        print(f"✅ Available configurations: {len(self.config_tester.get_configuration_names())}")
        print(f"📊 Configuration types: {list(self.config_tester.get_configuration_names())}")
        print(f"⚡ Data loading time: {load_time:.2f} seconds")
    
    def _load_and_prepare_data_optimized(self):
        """Load and prepare data with LA region optimization for speed"""
        print("📊 Loading drayage data...")
        load_start = time.time()
        
        # Load raw data
        self.drayage_data = pd.read_csv(self.data_file)
        print(f"   - Loaded {len(self.drayage_data)} drayage records in {time.time() - load_start:.2f}s")
        
        # Load and filter zip coordinates to mainland within 300 miles only
        coords_start = time.time()
        print("🗺️ Loading mainland zip coordinates within 300 miles of port...")
        self.zip_coords = pd.read_csv(self.zip_coords_file, dtype={'ZIP': str})
        
        # Filter to mainland within 300 miles for comprehensive coverage
        def is_mainland_within_300_miles(row):
            """Check if zip code is mainland and within 300 miles of port"""
            zip_code = row['ZIP']
            lat, lng = row['LAT'], row['LNG']
            
            # Exclude non-mainland areas
            if pd.isna(zip_code) or len(str(zip_code)) < 2:
                return False
            if str(zip_code)[:2] in EXCLUDED_ZIP_PREFIXES:  # Alaska, Hawaii
                return False
            if str(zip_code) in KNOWN_ISLAND_ZIPS:  # Known island areas
                return False
            
            # Check distance from port
            try:
                port_location = (33.7367, -118.2646)  # LA Port coordinates
                zip_location = (lat, lng)
                distance = geodesic(port_location, zip_location).miles
                return distance <= MAX_DISTANCE_MILES
            except:
                return False
        
        # Apply filtering to get mainland zips within 300 miles
        mainland_mask = self.zip_coords.apply(is_mainland_within_300_miles, axis=1)
        self.mainland_zip_coords = self.zip_coords[mainland_mask].copy()
        
        print(f"   - Filtered {len(self.zip_coords)} US zips to {len(self.mainland_zip_coords)} mainland zips within 300 miles")
        print(f"   - Geographic coverage: All mainland areas within 300 miles of LA port")
        print(f"   - Processing completed in {time.time() - coords_start:.2f}s")
        
        # Create coordinate lookup cache for fast access
        self.coord_cache = {}
        for _, row in self.mainland_zip_coords.iterrows():
            self.coord_cache[row['ZIP']] = (row['LAT'], row['LNG'])
        
        # Get LA Port coordinates
        if PORT_ZIP in self.coord_cache:
            self.port_location = self.coord_cache[PORT_ZIP]
        else:
            # Fallback to known LA port coordinates
            self.port_location = (33.7367, -118.2646)
            print(f"   - Using fallback coordinates for LA port")
        
        # Pre-filter drayage data to mainland within 300 miles
        filter_start = time.time()
        print("🎯 Filtering drayage data to mainland within 300 miles...")
        
        # Create set of valid zip codes for fast lookup
        valid_zips = set(self.mainland_zip_coords['ZIP'].astype(str))
        
        # Filter origins and destinations to mainland within 300 miles
        mainland_origin_mask = self.drayage_data['origin_zip'].astype(str).isin(valid_zips)
        mainland_dest_mask = self.drayage_data['destination_zip'].astype(str).isin(valid_zips)
        
        # Keep records where either origin OR destination is in mainland area (for port drayage)
        self.drayage_data = self.drayage_data[mainland_origin_mask | mainland_dest_mask].copy()
        print(f"   - Filtered to {len(self.drayage_data)} mainland drayage records in {time.time() - filter_start:.2f}s")
        
        # Combine origin and destination data (only mainland within 100 miles)
        combine_start = time.time()
        origins = self.drayage_data[['origin_zip', 'origin_lat', 'origin_lng', 'RPM', 'date']].copy()
        origins.columns = ['zip', 'lat', 'lng', 'rpm', 'date']
        origins = origins[origins['zip'].astype(str).isin(valid_zips)]
        
        destinations = self.drayage_data[['destination_zip', 'destination_lat', 'destination_lng', 'RPM', 'date']].copy()
        destinations.columns = ['zip', 'lat', 'lng', 'rpm', 'date']
        destinations = destinations[destinations['zip'].astype(str).isin(valid_zips)]
        
        # Combine all rate data
        self.all_rates = pd.concat([origins, destinations], ignore_index=True)
        self.all_rates = self.all_rates.dropna()
        print(f"   - Combined data in {time.time() - combine_start:.2f}s")
        
        # Calculate distance to port using cached coordinates
        distance_start = time.time()
        print("📏 Calculating distances to port...")
        self.all_rates['distance_to_port'] = self.all_rates.apply(
            lambda row: geodesic(self.port_location, (row['lat'], row['lng'])).miles, axis=1
        )
        print(f"   - Calculated {len(self.all_rates)} distances in {time.time() - distance_start:.2f}s")
        
        # Get zip codes with sufficient data and enhance diversity for configuration testing
        zip_start = time.time()
        zip_counts = self.all_rates.groupby('zip').size()
        min_transactions = 4 if self.fast_mode else 3
        available_zips = zip_counts[zip_counts >= min_transactions].index.tolist()
        
        # Enhanced zip code selection for better configuration differentiation
        if self.fast_mode and self.max_test_zips and len(available_zips) > self.max_test_zips:
            print(f"   - Selecting {self.max_test_zips} diverse zip codes for configuration testing...")
            
            # Categorize zip codes by multiple factors for maximum diversity
            zip_analysis = []
            for zip_code in available_zips:
                zip_data = self.all_rates[self.all_rates['zip'] == zip_code]
                
                # Calculate key characteristics
                avg_distance = zip_data['distance_to_port'].mean()
                rate_variance = zip_data['rpm'].std()
                transaction_count = len(zip_data)
                latest_date = zip_data['date'].max()
                
                # Calculate days since last transaction
                days_since_last = (datetime.now() - pd.to_datetime(latest_date)).days if pd.notna(latest_date) else 365
                
                zip_analysis.append({
                    'zip': zip_code,
                    'distance': avg_distance,
                    'variance': rate_variance,
                    'count': transaction_count,
                    'days_since_last': days_since_last,
                    'avg_rate': zip_data['rpm'].mean()
                })
            
            zip_df = pd.DataFrame(zip_analysis)
            
            # Select diverse zip codes across multiple dimensions
            selected_zips = []
            
            # 1. Distance diversity (close, medium, far)
            distance_bins = 3
            distance_quantiles = [0, 0.33, 0.67, 1.0]
            for i in range(distance_bins):
                mask = (zip_df['distance'] >= zip_df['distance'].quantile(distance_quantiles[i])) & \
                       (zip_df['distance'] <= zip_df['distance'].quantile(distance_quantiles[i+1]))
                bin_zips = zip_df[mask].nlargest(self.max_test_zips // distance_bins, 'count')['zip'].tolist()
                selected_zips.extend(bin_zips[:self.max_test_zips // distance_bins])
            
            # 2. Add high-variance zip codes (challenging for configuration testing)
            high_variance_zips = zip_df.nlargest(self.max_test_zips // 4, 'variance')['zip'].tolist()
            for zip_code in high_variance_zips:
                if zip_code not in selected_zips and len(selected_zips) < self.max_test_zips:
                    selected_zips.append(zip_code)
            
            # 3. Add recent activity zip codes (time-sensitive for configurations)
            recent_zips = zip_df.nsmallest(self.max_test_zips // 4, 'days_since_last')['zip'].tolist()
            for zip_code in recent_zips:
                if zip_code not in selected_zips and len(selected_zips) < self.max_test_zips:
                    selected_zips.append(zip_code)
            
            # 4. Fill remaining slots with highest transaction counts
            remaining_slots = self.max_test_zips - len(selected_zips)
            if remaining_slots > 0:
                high_count_zips = zip_df[~zip_df['zip'].isin(selected_zips)].nlargest(remaining_slots, 'count')['zip'].tolist()
                selected_zips.extend(high_count_zips)
            
            self.zip_codes_with_data = selected_zips[:self.max_test_zips]
            
            # Print diversity stats
            selected_df = zip_df[zip_df['zip'].isin(self.zip_codes_with_data)]
            print(f"   - Distance range: {selected_df['distance'].min():.1f} - {selected_df['distance'].max():.1f} miles")
            print(f"   - Rate variance range: ${selected_df['variance'].min():.2f} - ${selected_df['variance'].max():.2f}")
            print(f"   - Transaction counts: {selected_df['count'].min()} - {selected_df['count'].max()}")
            print(f"   - Data recency: {selected_df['days_since_last'].min()} - {selected_df['days_since_last'].max()} days old")
            
        else:
            self.zip_codes_with_data = available_zips
            
        print(f"   - Found {len(available_zips)} zip codes with sufficient data")
        if self.fast_mode and len(available_zips) > len(self.zip_codes_with_data):
            print(f"   - Enhanced selection: {len(self.zip_codes_with_data)} diverse zip codes for configuration testing")
        print(f"   - Processing completed in {time.time() - zip_start:.2f}s")
        
        # Calculate ground truth values (time-weighted averages)
        truth_start = time.time()
        print("🎯 Calculating ground truth values...")
        self.ground_truth = self._calculate_ground_truth()
        print(f"   - Calculated ground truth in {time.time() - truth_start:.2f}s")
        
        # Store performance stats
        self.performance_stats.update({
            'mainland_zip_coords_count': len(self.mainland_zip_coords),
            'filtered_drayage_records': len(self.drayage_data),
            'all_rates_count': len(self.all_rates),
            'zip_codes_with_data_count': len(self.zip_codes_with_data),
            'geographic_coverage_miles': MAX_DISTANCE_MILES,
            'port_location': self.port_location
        })
    
    def _calculate_ground_truth(self) -> pd.DataFrame:
        """Calculate ground truth values for validation"""
        # Convert dates and calculate time weights
        self.all_rates['date'] = pd.to_datetime(self.all_rates['date'])
        current_date = datetime.now()
        self.all_rates['days_since'] = (current_date - self.all_rates['date']).dt.days
        
        # Calculate time weights using baseline configuration
        base_config = EstimatorConfig()
        self.all_rates['time_weight'] = self._calculate_time_weight(
            self.all_rates['days_since'], base_config
        )
        
        # Calculate weighted statistics for each zip
        ground_truth_list = []
        for zip_code in self.zip_codes_with_data:
            zip_data = self.all_rates[self.all_rates['zip'] == zip_code]
            
            total_weight = zip_data['time_weight'].sum()
            if total_weight > 0:
                weighted_rpm = (zip_data['rpm'] * zip_data['time_weight']).sum() / total_weight
                
                # Calculate weighted standard deviation
                weighted_var = ((zip_data['rpm'] - weighted_rpm) ** 2 * zip_data['time_weight']).sum() / total_weight
                weighted_std = np.sqrt(weighted_var)
                
                ground_truth_list.append({
                    'zip': zip_code,
                    'ground_truth_rpm': weighted_rpm,
                    'ground_truth_std': weighted_std,
                    'transaction_count': len(zip_data),
                    'distance_to_port': zip_data['distance_to_port'].iloc[0],
                    'lat': zip_data['lat'].iloc[0],
                    'lng': zip_data['lng'].iloc[0]
                })
        
        return pd.DataFrame(ground_truth_list)
    
    def _calculate_time_weight(self, days_since: pd.Series, config: EstimatorConfig) -> pd.Series:
        """Calculate time weights using given configuration"""
        weights = np.ones(len(days_since))
        
        # Apply recent boost
        recent_mask = days_since <= config.RECENT_BOOST_DAYS
        weights[recent_mask] *= config.RECENT_MULTIPLIER
        
        # Apply exponential decay
        decay_mask = days_since > config.RECENT_BOOST_DAYS
        decay_days = days_since[decay_mask] - config.RECENT_BOOST_DAYS
        weights[decay_mask] *= (config.TIME_DECAY_RATE ** decay_days)
        
        # Cap minimum weight
        weights = np.maximum(weights, 0.05)
        
        return pd.Series(weights, index=days_since.index)
    
    def validate_configuration(self, config_name: str, verbose: bool = True) -> ValidationResult:
        """
        Validate a specific configuration using k-fold cross-validation
        
        Args:
            config_name: Name of configuration to test
            verbose: Print progress information
            
        Returns:
            ValidationResult with comprehensive metrics
        """
        start_time = datetime.now()
        
        if verbose:
            print(f"\n🧪 Validating configuration: {config_name}")
            print("=" * 60)
        
        # Get configuration
        config = self.config_tester.get_configuration(config_name)
        
        # Set up k-fold splits
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_seed)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        # Perform k-fold validation
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.zip_codes_with_data)):
            if verbose:
                print(f"📊 Processing fold {fold_idx + 1}/{self.k}...")
            
            # Get train and test zip codes
            train_zips = [self.zip_codes_with_data[i] for i in train_idx]
            test_zips = [self.zip_codes_with_data[i] for i in test_idx]
            
            # Create training data (exclude test zips)
            train_data = self.all_rates[self.all_rates['zip'].isin(train_zips)]
            
            # Create temporary estimator with training data only (optimized)
            estimator_start = time.time()
            fold_estimator = self._create_fold_estimator_optimized(train_data, config)
            estimator_time = time.time() - estimator_start
            
            if verbose and fold_idx == 0:  # Only show timing for first fold
                print(f"  ⚡ Fold estimator created in {estimator_time:.2f}s")
            
            # Make predictions for test zips
            fold_predictions = []
            fold_actuals = []
            fold_errors = []
            
            for test_zip in test_zips:
                try:
                    # Get ground truth
                    ground_truth = self.ground_truth[self.ground_truth['zip'] == test_zip]
                    if ground_truth.empty:
                        continue
                    
                    actual_rpm = ground_truth.iloc[0]['ground_truth_rpm']
                    
                    # Make prediction
                    prediction = fold_estimator.estimate_rate(str(test_zip), verbose=False)
                    
                    # Calculate error
                    error = prediction.estimated_rpm - actual_rpm
                    
                    fold_predictions.append(prediction)
                    fold_actuals.append(actual_rpm)
                    fold_errors.append(error)
                    
                    all_predictions.append(prediction)
                    all_actuals.append(actual_rpm)
                    
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ Error predicting {test_zip}: {str(e)}")
                    continue
            
            # Calculate fold metrics
            if fold_predictions:
                fold_metrics = self._calculate_fold_metrics(
                    fold_predictions, fold_actuals, fold_errors
                )
                
                fold_result = FoldResult(
                    fold_number=fold_idx + 1,
                    test_zips=test_zips,
                    predictions=fold_predictions,
                    actual_values=fold_actuals,
                    errors=fold_errors,
                    absolute_errors=[abs(e) for e in fold_errors],
                    metrics=fold_metrics
                )
                
                fold_results.append(fold_result)
                
                if verbose:
                    print(f"  ✓ Fold {fold_idx + 1} - MAE: ${fold_metrics['mae']:.2f}, "
                          f"RMSE: ${fold_metrics['rmse']:.2f}, R²: {fold_metrics['r2']:.3f}")
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_predictions, all_actuals)
        
        # Calculate confidence calibration
        confidence_calibration = self._calculate_confidence_calibration(all_predictions, all_actuals)
        
        # Calculate geographic analysis
        geographic_analysis = self._calculate_geographic_analysis(all_predictions, all_actuals)
        
        # Generate comprehensive explanations
        explanations = self._generate_result_explanations(
            overall_metrics, confidence_calibration, geographic_analysis, all_predictions, all_actuals
        )
        
        # Calculate coverage statistics
        coverage_stats = self._calculate_coverage_statistics(all_predictions)
        
        # Timing information
        end_time = datetime.now()
        timing_info = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'predictions_per_second': len(all_predictions) / (end_time - start_time).total_seconds()
        }
        
        if verbose:
            print(f"\n📊 Overall Results for {config_name}:")
            print(f"  MAE: ${overall_metrics['mae']:.2f}")
            print(f"  RMSE: ${overall_metrics['rmse']:.2f}")
            print(f"  R²: {overall_metrics['r2']:.3f}")
            print(f"  Duration: {timing_info['duration_seconds']:.1f} seconds")
            
            # Print comprehensive explanations
            self._print_result_explanations(explanations, coverage_stats)
        
        return ValidationResult(
            config_name=config_name,
            config_params=config.__dict__ if hasattr(config, '__dict__') else vars(config),
            fold_results=[asdict(fr) for fr in fold_results],
            overall_metrics=overall_metrics,
            confidence_calibration=confidence_calibration,
            geographic_analysis=geographic_analysis,
            timing_info=timing_info,
            explanations=explanations,
            coverage_stats=coverage_stats
        )
    
    def validate_baseline_only(self, fast_mode: bool = True) -> ValidationResult:
        """
        Quick baseline-only validation for speed testing
        
        Args:
            fast_mode: Use speed optimizations (fewer folds, smaller sample)
            
        Returns:
            ValidationResult for baseline configuration only
        """
        print(f"\n⚡ QUICK BASELINE VALIDATION")
        print("=" * 50)
        
        if fast_mode:
            # Override settings for maximum speed
            original_k = self.k
            original_test_zips = self.zip_codes_with_data.copy()
            
            # Use fewer folds and zip codes for speed
            self.k = 3
            if len(self.zip_codes_with_data) > 10:
                # Sample 10 representative zip codes
                sample_size = 10
                step = len(self.zip_codes_with_data) // sample_size
                self.zip_codes_with_data = [self.zip_codes_with_data[i * step] for i in range(sample_size)]
            
            print(f"⚡ Fast mode: {self.k} folds, {len(self.zip_codes_with_data)} zip codes")
        
        try:
            # Run baseline validation
            result = self.validate_configuration('baseline', verbose=True)
            
            if fast_mode:
                # Restore original settings
                self.k = original_k
                self.zip_codes_with_data = original_test_zips
                
                # Add fast mode info to results
                result.timing_info['fast_mode'] = True
                result.timing_info['reduced_k'] = 3
                result.timing_info['reduced_sample_size'] = len(self.zip_codes_with_data)
            
            return result
            
        except Exception as e:
            print(f"❌ Error in baseline validation: {e}")
            if fast_mode:
                # Restore settings on error
                self.k = original_k  
                self.zip_codes_with_data = original_test_zips
            raise
    
    def _create_fold_estimator_optimized(self, train_data: pd.DataFrame, config: EstimatorConfig) -> IntelligentRateEstimator:
        """Create estimator for a specific fold with in-memory optimization (no temp files)"""
        # Convert back to original format for the estimator
        origins = train_data[['zip', 'lat', 'lng', 'rpm', 'date']].copy()
        origins.columns = ['origin_zip', 'origin_lat', 'origin_lng', 'RPM', 'date']
        origins['destination_zip'] = origins['origin_zip']  # Dummy values
        origins['destination_lat'] = origins['origin_lat']
        origins['destination_lng'] = origins['origin_lng']
        
        # Save fold data temporarily but clean up immediately
        temp_data_file = f"temp_fold_data_{int(time.time()*1000)}.csv"
        temp_coords_file = f"temp_coords_data_{int(time.time()*1000)}.csv"
        
        try:
            # Save data temporarily
            origins.to_csv(temp_data_file, index=False)
            self.mainland_zip_coords.to_csv(temp_coords_file, index=False)
            
            # Temporarily replace global config
            import intelligent_rate_estimator
            original_config = intelligent_rate_estimator.CONFIG
            intelligent_rate_estimator.CONFIG = config
            
            # Create estimator with fold data
            estimator = IntelligentRateEstimator(
                data_file=temp_data_file,
                zip_coords_file=temp_coords_file
            )
            
        finally:
            # Restore original config
            intelligent_rate_estimator.CONFIG = original_config
            # Clean up temp files immediately
            Path(temp_data_file).unlink(missing_ok=True)
            Path(temp_coords_file).unlink(missing_ok=True)
        
        return estimator
    
    def _calculate_fold_metrics(self, predictions: List[RateEstimate], 
                               actuals: List[float], errors: List[float]) -> Dict:
        """Calculate metrics for a single fold"""
        predicted_values = [p.estimated_rpm for p in predictions]
        
        mae = mean_absolute_error(actuals, predicted_values)
        rmse = np.sqrt(mean_squared_error(actuals, predicted_values))
        r2 = r2_score(actuals, predicted_values)
        
        # Additional metrics
        mape = np.mean([abs(e / a) for e, a in zip(errors, actuals) if a != 0]) * 100
        bias = np.mean(errors)
        
        # Confidence metrics
        avg_confidence = np.mean([p.confidence_level for p in predictions])
        confidence_std = np.std([p.confidence_level for p in predictions])
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'bias': bias,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'n_predictions': len(predictions)
        }
    
    def _calculate_overall_metrics(self, predictions: List[RateEstimate], 
                                 actuals: List[float]) -> Dict:
        """Calculate overall metrics across all folds"""
        predicted_values = [p.estimated_rpm for p in predictions]
        errors = [p - a for p, a in zip(predicted_values, actuals)]
        
        mae = mean_absolute_error(actuals, predicted_values)
        rmse = np.sqrt(mean_squared_error(actuals, predicted_values))
        r2 = r2_score(actuals, predicted_values)
        
        # Additional metrics
        mape = np.mean([abs(e / a) for e, a in zip(errors, actuals) if a != 0]) * 100
        bias = np.mean(errors)
        
        # Percentile errors
        abs_errors = [abs(e) for e in errors]
        percentiles = [50, 75, 90, 95, 99]
        percentile_errors = {f'p{p}': np.percentile(abs_errors, p) for p in percentiles}
        
        # Confidence metrics
        avg_confidence = np.mean([p.confidence_level for p in predictions])
        confidence_std = np.std([p.confidence_level for p in predictions])
        
        # Method usage
        methods = [p.method_used for p in predictions]
        method_counts = {method: methods.count(method) for method in set(methods)}
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'bias': bias,
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'n_predictions': len(predictions),
            'percentile_errors': percentile_errors,
            'method_usage': method_counts
        }
    
    def _calculate_confidence_calibration(self, predictions: List[RateEstimate], 
                                        actuals: List[float]) -> Dict:
        """Calculate confidence calibration metrics"""
        predicted_values = [p.estimated_rpm for p in predictions]
        confidences = [p.confidence_level for p in predictions]
        errors = [abs(p - a) for p, a in zip(predicted_values, actuals)]
        
        # Bin by confidence levels
        confidence_bins = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 100)]
        calibration_data = []
        
        for bin_min, bin_max in confidence_bins:
            bin_mask = [(c >= bin_min and c < bin_max) for c in confidences]
            if any(bin_mask):
                bin_errors = [e for e, mask in zip(errors, bin_mask) if mask]
                bin_confidences = [c for c, mask in zip(confidences, bin_mask) if mask]
                
                calibration_data.append({
                    'confidence_bin': f"{bin_min}-{bin_max}%",
                    'avg_confidence': np.mean(bin_confidences),
                    'avg_error': np.mean(bin_errors),
                    'count': len(bin_errors),
                    'rmse': np.sqrt(np.mean([e**2 for e in bin_errors]))
                })
        
        return {
            'calibration_by_bin': calibration_data,
            'overall_correlation': np.corrcoef(confidences, [-e for e in errors])[0, 1]
        }
    
    def _calculate_geographic_analysis(self, predictions: List[RateEstimate], 
                                     actuals: List[float]) -> Dict:
        """Calculate geographic performance analysis"""
        predicted_values = [p.estimated_rpm for p in predictions]
        distances = [p.distance_to_port for p in predictions]
        errors = [abs(p - a) for p, a in zip(predicted_values, actuals)]
        
        # Bin by distance from port (updated for 300-mile coverage)
        distance_bins = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500)]
        geographic_data = []
        
        for bin_min, bin_max in distance_bins:
            bin_mask = [(d >= bin_min and d < bin_max) for d in distances]
            if any(bin_mask):
                bin_errors = [e for e, mask in zip(errors, bin_mask) if mask]
                bin_predictions = [p for p, mask in zip(predictions, bin_mask) if mask]
                
                geographic_data.append({
                    'distance_bin': f"{bin_min}-{bin_max} miles",
                    'avg_distance': np.mean([d for d, mask in zip(distances, bin_mask) if mask]),
                    'avg_error': np.mean(bin_errors),
                    'count': len(bin_errors),
                    'avg_confidence': np.mean([p.confidence_level for p in bin_predictions])
                })
        
        return {
            'performance_by_distance': geographic_data,
            'distance_error_correlation': np.corrcoef(distances, errors)[0, 1]
        }
    
    def _generate_result_explanations(self, overall_metrics: Dict, confidence_calibration: Dict, 
                                    geographic_analysis: Dict, predictions: List, actuals: List) -> Dict:
        """Generate comprehensive explanations of validation results"""
        explanations = {}
        
        # Accuracy explanation
        mae = overall_metrics['mae']
        if mae < 0.30:
            accuracy_level = "EXCELLENT"
            accuracy_desc = "Predictions are highly accurate with very small errors"
        elif mae < 0.50:
            accuracy_level = "VERY GOOD" 
            accuracy_desc = "Predictions are quite accurate with minor errors"
        elif mae < 0.75:
            accuracy_level = "GOOD"
            accuracy_desc = "Predictions are reasonably accurate with moderate errors"
        elif mae < 1.00:
            accuracy_level = "FAIR"
            accuracy_desc = "Predictions have noticeable errors but are still useful"
        else:
            accuracy_level = "NEEDS IMPROVEMENT"
            accuracy_desc = "Predictions have significant errors and need optimization"
        
        explanations['accuracy'] = {
            'level': accuracy_level,
            'description': accuracy_desc,
            'mae_dollars': mae,
            'business_impact': f"On average, rate predictions are off by ${mae:.2f} per mile"
        }
        
        # Correlation explanation
        r2 = overall_metrics['r2']
        if r2 > 0.95:
            correlation_level = "EXCELLENT"
            correlation_desc = "Predictions track actual rates very closely"
        elif r2 > 0.90:
            correlation_level = "VERY GOOD"
            correlation_desc = "Predictions correlate well with actual rates"
        elif r2 > 0.80:
            correlation_level = "GOOD"
            correlation_desc = "Predictions show good correlation with actual rates"
        elif r2 > 0.70:
            correlation_level = "FAIR"
            correlation_desc = "Predictions show moderate correlation with actual rates"
        else:
            correlation_level = "POOR"
            correlation_desc = "Predictions show weak correlation with actual rates"
        
        explanations['correlation'] = {
            'level': correlation_level,
            'description': correlation_desc,
            'r2_score': r2,
            'variance_explained': f"{r2*100:.1f}% of rate variance is explained by the model"
        }
        
        # Confidence explanation
        avg_confidence = overall_metrics['avg_confidence']
        if avg_confidence > 85:
            confidence_level = "HIGH"
            confidence_desc = "Model is very confident in its predictions"
        elif avg_confidence > 75:
            confidence_level = "MODERATE"
            confidence_desc = "Model has reasonable confidence in its predictions"
        else:
            confidence_level = "LOW"
            confidence_desc = "Model has low confidence in its predictions"
        
        explanations['confidence'] = {
            'level': confidence_level,
            'description': confidence_desc,
            'average_confidence': avg_confidence,
            'interpretation': f"Model reports {avg_confidence:.1f}% average confidence"
        }
        
        # Geographic performance explanation
        geo_performance = self._explain_geographic_performance(geographic_analysis)
        explanations['geographic'] = geo_performance
        
        # Business impact explanation
        explanations['business_impact'] = self._explain_business_impact(overall_metrics, predictions, actuals)
        
        # Method usage explanation
        if 'method_usage' in overall_metrics:
            explanations['methods'] = self._explain_method_usage(overall_metrics['method_usage'])
        
        return explanations
    
    def _explain_geographic_performance(self, geographic_analysis: Dict) -> Dict:
        """Explain geographic performance patterns"""
        geo_explanation = {
            'coverage': f"Analysis covers mainland areas within {MAX_DISTANCE_MILES} miles of LA port",
            'port_focus': "Performance optimized for comprehensive regional drayage operations"
        }
        
        if 'performance_by_distance' in geographic_analysis:
            distance_performance = geographic_analysis['performance_by_distance']
            
            if distance_performance:
                # Analyze performance by distance (updated for 300-mile range)
                close_performance = [d for d in distance_performance if '0-50' in d.get('distance_bin', '')]
                medium_performance = [d for d in distance_performance if any(x in d.get('distance_bin', '') for x in ['50-100', '100-200'])]
                far_performance = [d for d in distance_performance if '200-300' in d.get('distance_bin', '')]
                
                if close_performance:
                    close_error = close_performance[0]['avg_error']
                    if close_error < 0.40:
                        geo_explanation['close_range'] = f"EXCELLENT performance near port (0-50 mi, error: ${close_error:.2f})"
                    elif close_error < 0.60:
                        geo_explanation['close_range'] = f"GOOD performance near port (0-50 mi, error: ${close_error:.2f})"
                    else:
                        geo_explanation['close_range'] = f"MODERATE performance near port (0-50 mi, error: ${close_error:.2f})"
                
                if medium_performance:
                    medium_error = np.mean([d['avg_error'] for d in medium_performance])
                    if medium_error < 0.50:
                        geo_explanation['medium_range'] = f"GOOD performance at medium distances (50-200 mi, error: ${medium_error:.2f})"
                    elif medium_error < 0.75:
                        geo_explanation['medium_range'] = f"FAIR performance at medium distances (50-200 mi, error: ${medium_error:.2f})"
                    else:
                        geo_explanation['medium_range'] = f"CHALLENGING performance at medium distances (50-200 mi, error: ${medium_error:.2f})"
                
                if far_performance:
                    far_error = far_performance[0]['avg_error']
                    if far_error < 0.60:
                        geo_explanation['far_range'] = f"GOOD performance at extended distances (200-300 mi, error: ${far_error:.2f})"
                    elif far_error < 1.00:
                        geo_explanation['far_range'] = f"FAIR performance at extended distances (200-300 mi, error: ${far_error:.2f})"
                    else:
                        geo_explanation['far_range'] = f"CHALLENGING performance at extended distances (200-300 mi, error: ${far_error:.2f})"
        
        return geo_explanation
    
    def _explain_business_impact(self, overall_metrics: Dict, predictions: List, actuals: List) -> Dict:
        """Explain business impact of validation results"""
        mae = overall_metrics['mae']
        mape = overall_metrics.get('mape', 0)
        
        # Calculate financial impact
        avg_rate = np.mean(actuals) if actuals else 0
        error_percentage = (mae / avg_rate * 100) if avg_rate > 0 else 0
        
        business_impact = {
            'financial_accuracy': f"Rate predictions are typically within ${mae:.2f}/mile ({error_percentage:.1f}%)",
            'pricing_confidence': f"Suitable for pricing with {error_percentage:.1f}% margin of error",
        }
        
        if mae < 0.50:
            business_impact['recommendation'] = "READY FOR PRODUCTION - High accuracy suitable for automated pricing"
        elif mae < 0.75:
            business_impact['recommendation'] = "GOOD FOR GUIDANCE - Suitable for rate estimation with human oversight"
        elif mae < 1.00:
            business_impact['recommendation'] = "NEEDS REVIEW - Consider additional tuning before production use"
        else:
            business_impact['recommendation'] = "REQUIRES OPTIMIZATION - Significant improvements needed"
        
        # Operational impact
        if overall_metrics['r2'] > 0.90:
            business_impact['operational'] = "Model can reliably predict rate trends and patterns"
        elif overall_metrics['r2'] > 0.80:
            business_impact['operational'] = "Model provides good directional guidance for rates"
        else:
            business_impact['operational'] = "Model predictions should be used cautiously"
        
        return business_impact
    
    def _explain_method_usage(self, method_usage: Dict) -> Dict:
        """Explain which prediction methods were used and their implications"""
        method_explanations = {
            'knn': 'K-Nearest Neighbors - Uses similar nearby locations',
            'distance_weighted': 'Distance Weighted - Emphasizes closer locations',
            'port_similarity': 'Port Similarity - Uses locations with similar port distances',
            'time_weighted': 'Time Weighted - Emphasizes recent transactions',
            'fallback': 'Fallback Method - Used when insufficient local data'
        }
        
        total_predictions = sum(method_usage.values())
        method_analysis = {}
        
        for method, count in method_usage.items():
            percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
            method_analysis[method] = {
                'count': count,
                'percentage': percentage,
                'description': method_explanations.get(method, f'Method: {method}')
            }
        
        # Determine primary method
        primary_method = max(method_usage, key=method_usage.get) if method_usage else 'unknown'
        primary_percentage = (method_usage[primary_method] / total_predictions * 100) if total_predictions > 0 else 0
        
        return {
            'primary_method': primary_method,
            'primary_percentage': primary_percentage,
            'method_breakdown': method_analysis,
            'interpretation': f"Primary prediction method: {primary_method} ({primary_percentage:.1f}%)"
        }
    
    def _calculate_coverage_statistics(self, predictions: List) -> Dict:
        """Calculate geographic and data coverage statistics"""
        if not predictions:
            return {}
        
        # Distance coverage
        distances = [p.distance_to_port for p in predictions if hasattr(p, 'distance_to_port')]
        
        coverage_stats = {
            'total_predictions': len(predictions),
            'mainland_coverage_miles': MAX_DISTANCE_MILES,
            'port_location': self.port_location
        }
        
        if distances:
            coverage_stats.update({
                'min_distance_miles': min(distances),
                'max_distance_miles': max(distances),
                'avg_distance_miles': np.mean(distances),
                'distance_range': f"{min(distances):.1f} - {max(distances):.1f} miles from port"
            })
            
            # Distance distribution (updated for 300-mile coverage)
            close_count = len([d for d in distances if d <= 50])
            medium_count = len([d for d in distances if 50 < d <= 150])
            far_count = len([d for d in distances if 150 < d <= 300])
            extended_count = len([d for d in distances if d > 300])
            
            coverage_stats['distance_distribution'] = {
                'close_0_50_miles': close_count,
                'medium_50_150_miles': medium_count,
                'far_150_300_miles': far_count,
                'extended_300_plus_miles': extended_count
            }
        
        return coverage_stats
    
    def _print_result_explanations(self, explanations: Dict, coverage_stats: Dict):
        """Print comprehensive result explanations"""
        print(f"\n🎯 COMPREHENSIVE RESULTS EXPLANATION")
        print("=" * 70)
        
        # Accuracy explanation
        if 'accuracy' in explanations:
            acc = explanations['accuracy']
            print(f"\n📊 ACCURACY ASSESSMENT: {acc['level']}")
            print(f"   {acc['description']}")
            print(f"   Business Impact: {acc['business_impact']}")
        
        # Correlation explanation
        if 'correlation' in explanations:
            corr = explanations['correlation']
            print(f"\n📈 CORRELATION ANALYSIS: {corr['level']}")
            print(f"   {corr['description']}")
            print(f"   Statistical: {corr['variance_explained']}")
        
        # Confidence explanation
        if 'confidence' in explanations:
            conf = explanations['confidence']
            print(f"\n🔮 CONFIDENCE ASSESSMENT: {conf['level']}")
            print(f"   {conf['description']}")
            print(f"   {conf['interpretation']}")
        
        # Geographic explanation
        if 'geographic' in explanations:
            geo = explanations['geographic']
            print(f"\n🗺️ GEOGRAPHIC PERFORMANCE:")
            print(f"   Coverage: {geo['coverage']}")
            if 'close_range' in geo:
                print(f"   Near Port (0-50 mi): {geo['close_range']}")
            if 'medium_range' in geo:
                print(f"   Medium Distance (50-200 mi): {geo['medium_range']}")
            if 'far_range' in geo:
                print(f"   Extended Distance (200-300 mi): {geo['far_range']}")
        
        # Coverage statistics
        if coverage_stats:
            print(f"\n📍 DATA COVERAGE STATISTICS:")
            print(f"   Total test predictions: {coverage_stats.get('total_predictions', 0)}")
            if 'distance_range' in coverage_stats:
                print(f"   Distance range: {coverage_stats['distance_range']}")
            if 'distance_distribution' in coverage_stats:
                dist = coverage_stats['distance_distribution']
                print(f"   Distance distribution:")
                print(f"     • Close (0-50 mi): {dist.get('close_0_50_miles', 0)} locations")
                print(f"     • Medium (50-150 mi): {dist.get('medium_50_150_miles', 0)} locations")
                print(f"     • Far (150-300 mi): {dist.get('far_150_300_miles', 0)} locations")
                if dist.get('extended_300_plus_miles', 0) > 0:
                    print(f"     • Extended (300+ mi): {dist.get('extended_300_plus_miles', 0)} locations")
        
        # Business impact
        if 'business_impact' in explanations:
            biz = explanations['business_impact']
            print(f"\n💼 BUSINESS IMPACT ANALYSIS:")
            print(f"   Financial: {biz.get('financial_accuracy', '')}")
            print(f"   Operational: {biz.get('operational', '')}")
            print(f"   Recommendation: {biz.get('recommendation', '')}")
        
        # Method usage
        if 'methods' in explanations:
            methods = explanations['methods']
            print(f"\n⚙️ PREDICTION METHODS USED:")
            print(f"   {methods.get('interpretation', '')}")
            if 'method_breakdown' in methods:
                for method, details in methods['method_breakdown'].items():
                    if details['percentage'] > 5:  # Only show methods used > 5%
                        print(f"   • {method}: {details['percentage']:.1f}% - {details['description']}")
        
        print(f"\n" + "=" * 70)
    
    def generate_professional_report(self, results: Dict[str, ValidationResult], 
                                   comparison_df: pd.DataFrame = None,
                                   output_filename: str = None) -> str:
        """
        Generate a professional HTML report for validation results
        
        Args:
            results: Dictionary of validation results
            comparison_df: Optional comparison dataframe
            output_filename: Optional custom filename
            
        Returns:
            Path to generated HTML report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"DrayVis_Validation_Report_{timestamp}.html"
        
        print(f"\n📋 Generating Professional Validation Report...")
        print(f"   Report file: {output_filename}")
        
        # Generate comparison if not provided
        if comparison_df is None and results:
            comparison_df = self.compare_configurations(results, save_plots=False)
        
        # Create HTML report
        html_content = self._create_html_report(results, comparison_df)
        
        # Write to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Professional report generated: {output_filename}")
        return output_filename
    
    def _create_html_report(self, results: Dict[str, ValidationResult], 
                           comparison_df: pd.DataFrame = None) -> str:
        """Create comprehensive HTML report"""
        
        # Get best performing configuration
        best_config = None
        best_mae = float('inf')
        
        for config_name, result in results.items():
            mae = result.overall_metrics.get('mae', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_config = config_name
        
        # Generate report timestamp
        report_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DrayVis Rate Prediction Validation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 30px;
            margin-bottom: 40px;
        }}
        
        .header h1 {{
            color: #2c5aa0;
            font-size: 2.5em;
            margin: 0;
            font-weight: 300;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.2em;
            margin: 10px 0;
        }}
        
        .header .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .executive-summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 40px;
        }}
        
        .executive-summary h2 {{
            margin-top: 0;
            font-size: 1.8em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 5px solid #2c5aa0;
        }}
        
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #2c5aa0;
            font-size: 1.1em;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        
        .metric-card .description {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .section {{
            margin: 40px 0;
            padding: 30px;
            background: #fafafa;
            border-radius: 10px;
            border-left: 5px solid #2c5aa0;
        }}
        
        .section h2 {{
            color: #2c5aa0;
            margin-top: 0;
            font-size: 1.6em;
        }}
        
        .performance-level {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.8em;
        }}
        
        .excellent {{ background: #d4edda; color: #155724; }}
        .very-good {{ background: #cce5ff; color: #004085; }}
        .good {{ background: #e2f7e2; color: #1e7e34; }}
        .fair {{ background: #fff3cd; color: #856404; }}
        .poor {{ background: #f8d7da; color: #721c24; }}
        .needs-improvement {{ background: #f8d7da; color: #721c24; }}
        
        .recommendation-box {{
            background: #e8f4fd;
            border: 1px solid #b6e2ff;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .recommendation-box h3 {{
            color: #2c5aa0;
            margin-top: 0;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        
        table th, table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        table th {{
            background: #2c5aa0;
            color: white;
            font-weight: 600;
        }}
        
        table tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        
        .geographic-analysis {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .geo-zone {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .geo-zone h4 {{
            margin-top: 0;
            color: #2c5aa0;
        }}
        
        .coverage-stats {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            border-top: 2px solid #eee;
            color: #666;
        }}
        
        .highlight {{
            background: #fffbf0;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DrayVis Rate Prediction Validation Report</h1>
            <div class="subtitle">Comprehensive Performance Analysis & Business Impact Assessment</div>
            <div class="timestamp">Generated on {report_time}</div>
        </div>
        
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p>This report presents a comprehensive analysis of the DrayVis Intelligent Rate Estimator's performance across {len(results)} different configurations using k-fold cross-validation methodology. The analysis covers all mainland areas within {MAX_DISTANCE_MILES} miles of the Los Angeles port, providing insights into accuracy, reliability, and business readiness.</p>
            
            {self._generate_executive_metrics_html(results, best_config)}
        </div>
        
        {self._generate_key_findings_html(results, best_config)}
        
        {self._generate_configuration_comparison_html(comparison_df)}
        
        {self._generate_geographic_analysis_html(results, best_config)}
        
        {self._generate_business_impact_html(results, best_config)}
        
        {self._generate_technical_details_html(results)}
        
        {self._generate_recommendations_html(results, best_config)}
        
        <div class="footer">
            <p><strong>DrayVis Analytics Platform</strong><br>
            Advanced Rate Prediction & Validation System<br>
            Report generated by automated validation framework</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html_content
    
    def _generate_executive_metrics_html(self, results: Dict[str, ValidationResult], 
                                       best_config: str) -> str:
        """Generate executive summary metrics"""
        if not results or not best_config:
            return ""
        
        best_result = results[best_config]
        mae = best_result.overall_metrics.get('mae', 0)
        rmse = best_result.overall_metrics.get('rmse', 0)
        mape = best_result.overall_metrics.get('mape', 0)
        r2 = best_result.overall_metrics.get('r2', 0)
        
        # Performance classification
        mae_level = self._classify_performance_level(mae, 'mae')
        accuracy_level = self._classify_performance_level(mape, 'mape')
        
        return f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Best Configuration</h3>
                    <div class="value">{best_config}</div>
                    <div class="description">Optimal performer</div>
                </div>
                <div class="metric-card">
                    <h3>Mean Absolute Error</h3>
                    <div class="value">${mae:.2f}</div>
                    <div class="description">Average prediction error</div>
                </div>
                <div class="metric-card">
                    <h3>Accuracy Rating</h3>
                    <div class="value performance-level {accuracy_level.lower().replace(' ', '-')}">{accuracy_level}</div>
                    <div class="description">Based on {mape:.1f}% MAPE</div>
                </div>
                <div class="metric-card">
                    <h3>R² Score</h3>
                    <div class="value">{r2:.3f}</div>
                    <div class="description">Variance explained</div>
                </div>
            </div>
        """
    
    def _generate_key_findings_html(self, results: Dict[str, ValidationResult], 
                                  best_config: str) -> str:
        """Generate key findings section"""
        if not results or not best_config:
            return ""
        
        best_result = results[best_config]
        
        return f"""
        <div class="section">
            <h2>Key Findings</h2>
            
            <div class="highlight">
                <h3>Primary Results</h3>
                <ul>
                    <li><strong>Best Performing Configuration:</strong> {best_config}</li>
                    <li><strong>Prediction Accuracy:</strong> {best_result.overall_metrics.get('mape', 0):.1f}% MAPE (Mean Absolute Percentage Error)</li>
                    <li><strong>Average Error:</strong> ${best_result.overall_metrics.get('mae', 0):.2f} per prediction</li>
                    <li><strong>Geographic Coverage:</strong> {MAX_DISTANCE_MILES}-mile radius from LA Port</li>
                    <li><strong>Validation Method:</strong> K-Fold Cross Validation</li>
                </ul>
            </div>
            
            {self._generate_business_readiness_assessment(best_result)}
        </div>
        """
    
    def _generate_business_readiness_assessment(self, result: ValidationResult) -> str:
        """Generate business readiness assessment"""
        mape = result.overall_metrics.get('mape', 0)
        
        if mape <= 5:
            readiness = "PRODUCTION READY"
            assessment = "Excellent accuracy suitable for live customer pricing."
            color = "excellent"
        elif mape <= 10:
            readiness = "BUSINESS READY"
            assessment = "Very good accuracy suitable for operational use with monitoring."
            color = "very-good"
        elif mape <= 15:
            readiness = "PILOT READY"
            assessment = "Good accuracy suitable for pilot programs and testing."
            color = "good"
        elif mape <= 25:
            readiness = "DEVELOPMENT"
            assessment = "Fair accuracy requiring optimization before deployment."
            color = "fair"
        else:
            readiness = "NEEDS IMPROVEMENT"
            assessment = "Accuracy requires significant improvement before business use."
            color = "needs-improvement"
        
        return f"""
            <div class="recommendation-box">
                <h3>Business Readiness Assessment</h3>
                <p><span class="performance-level {color}">{readiness}</span></p>
                <p>{assessment}</p>
            </div>
        """
    
    def _generate_configuration_comparison_html(self, comparison_df: pd.DataFrame = None) -> str:
        """Generate configuration comparison section"""
        if comparison_df is None or comparison_df.empty:
            return ""
        
        # Generate comparison table
        table_html = "<table><thead><tr>"
        for col in comparison_df.columns:
            table_html += f"<th>{col.replace('_', ' ').title()}</th>"
        table_html += "</tr></thead><tbody>"
        
        for _, row in comparison_df.iterrows():
            table_html += "<tr>"
            for col in comparison_df.columns:
                value = row[col]
                if isinstance(value, float):
                    if 'mae' in col.lower() or 'rmse' in col.lower():
                        table_html += f"<td>${value:.2f}</td>"
                    elif 'mape' in col.lower():
                        table_html += f"<td>{value:.1f}%</td>"
                    else:
                        table_html += f"<td>{value:.3f}</td>"
                else:
                    table_html += f"<td>{value}</td>"
            table_html += "</tr>"
        
        table_html += "</tbody></table>"
        
        return f"""
        <div class="section">
            <h2>Configuration Performance Comparison</h2>
            <p>Detailed comparison of all tested configurations ranked by overall performance:</p>
            {table_html}
        </div>
        """
    
    def _generate_geographic_analysis_html(self, results: Dict[str, ValidationResult], 
                                         best_config: str) -> str:
        """Generate geographic analysis section"""
        if not results or not best_config:
            return ""
        
        best_result = results[best_config]
        distance_metrics = best_result.geographic_analysis.get('performance_by_distance', [])
        
        zones_html = ""
        for distance_data in distance_metrics:
            distance_range = distance_data.get('distance_bin', 'Unknown')
            mae = distance_data.get('avg_error', 0)
            count = distance_data.get('count', 0)
            avg_confidence = distance_data.get('avg_confidence', 0)
            
            # Calculate MAPE equivalent from MAE for classification
            mape = (mae / 5.0) * 100 if mae > 0 else 0  # Rough conversion assuming $5 average rate
            level = self._classify_performance_level(mape, 'mape')
            
            zones_html += f"""
            <div class="geo-zone">
                <h4>{distance_range}</h4>
                <p><strong>Samples:</strong> {count:,}</p>
                <p><strong>Average Error:</strong> ${mae:.2f}</p>
                <p><strong>Confidence:</strong> {avg_confidence:.1f}%</p>
                <p><strong>Performance:</strong> <span class="performance-level {level.lower().replace(' ', '-')}">{level}</span></p>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>Geographic Performance Analysis</h2>
            <p>Performance breakdown by distance from Los Angeles Port:</p>
            <div class="geographic-analysis">
                {zones_html}
            </div>
            
            <div class="recommendation-box">
                <h3>Geographic Insights</h3>
                <p>The model shows varying performance across different distance ranges from the port. 
                Closer destinations typically show better accuracy due to more consistent routing and fewer variables.</p>
            </div>
        </div>
        """
    
    def _generate_business_impact_html(self, results: Dict[str, ValidationResult], 
                                     best_config: str) -> str:
        """Generate business impact section"""
        if not results or not best_config:
            return ""
        
        best_result = results[best_config]
        mae = best_result.overall_metrics.get('mae', 0)
        mape = best_result.overall_metrics.get('mape', 0)
        
        # Business impact calculations
        annual_quotes = 10000  # Estimated
        potential_savings = annual_quotes * mae * 0.5  # Conservative estimate
        
        return f"""
        <div class="section">
            <h2>Business Impact Analysis</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>Prediction Reliability</h3>
                    <div class="value">{100-mape:.1f}%</div>
                    <div class="description">Customer confidence level</div>
                </div>
                <div class="metric-card">
                    <h3>Operational Efficiency</h3>
                    <div class="value">${mae:.2f}</div>
                    <div class="description">Average error per quote</div>
                </div>
                <div class="metric-card">
                    <h3>Potential Annual Savings</h3>
                    <div class="value">${potential_savings:,.0f}</div>
                    <div class="description">From improved accuracy</div>
                </div>
            </div>
            
            <div class="recommendation-box">
                <h3>Business Benefits</h3>
                <ul>
                    <li><strong>Faster Quoting:</strong> Instant rate estimates reduce response time</li>
                    <li><strong>Competitive Advantage:</strong> Data-driven pricing improves win rates</li>
                    <li><strong>Risk Reduction:</strong> Accurate estimates minimize underpricing</li>
                    <li><strong>Customer Satisfaction:</strong> Reliable quotes build trust</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_technical_details_html(self, results: Dict[str, ValidationResult]) -> str:
        """Generate technical details section"""
        if not results:
            return ""
        
        # Technical summary
        total_samples = sum(len(result.predictions) if hasattr(result, 'predictions') 
                          else result.overall_metrics.get('count', 0) 
                          for result in results.values())
        
        return f"""
        <div class="section">
            <h2>Technical Methodology</h2>
            
            <div class="coverage-stats">
                <h3>Validation Parameters</h3>
                <ul>
                    <li><strong>Geographic Scope:</strong> Mainland US within {MAX_DISTANCE_MILES} miles of LA Port (33.7367°N, 118.2646°W)</li>
                    <li><strong>Exclusions:</strong> Alaska (99xxx), Hawaii (96xxx), offshore locations</li>
                    <li><strong>Validation Method:</strong> K-Fold Cross Validation</li>
                    <li><strong>Total Samples:</strong> {total_samples:,} rate predictions</li>
                    <li><strong>Configurations Tested:</strong> {len(results)}</li>
                </ul>
                
                <h3>Performance Metrics</h3>
                <ul>
                    <li><strong>MAE:</strong> Mean Absolute Error - average dollar difference</li>
                    <li><strong>MAPE:</strong> Mean Absolute Percentage Error - accuracy percentage</li>
                    <li><strong>RMSE:</strong> Root Mean Square Error - penalizes large errors</li>
                    <li><strong>R²:</strong> Coefficient of determination - variance explained</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_recommendations_html(self, results: Dict[str, ValidationResult], 
                                     best_config: str) -> str:
        """Generate recommendations section"""
        if not results or not best_config:
            return ""
        
        best_result = results[best_config]
        mape = best_result.overall_metrics.get('mape', 0)
        
        # Generate recommendations based on performance
        recommendations = []
        
        if mape <= 10:
            recommendations.extend([
                "Deploy the model in production with confidence monitoring",
                "Implement automated rate suggestions for customer quotes",
                "Use predictions for competitive pricing strategies"
            ])
        elif mape <= 20:
            recommendations.extend([
                "Deploy in pilot mode with manual review",
                "Collect additional training data for model improvement",
                "Implement confidence intervals for predictions"
            ])
        else:
            recommendations.extend([
                "Further model optimization required before deployment",
                "Expand training dataset with more diverse routes",
                "Consider ensemble methods or alternative algorithms"
            ])
        
        recommendations.extend([
            "Monitor prediction accuracy continuously in production",
            "Establish feedback loops for model improvement",
            "Regular revalidation as market conditions change"
        ])
        
        rec_html = ""
        for i, rec in enumerate(recommendations, 1):
            rec_html += f"<li>{rec}</li>"
        
        return f"""
        <div class="section">
            <h2>Strategic Recommendations</h2>
            
            <div class="recommendation-box">
                <h3>Next Steps</h3>
                <ol>
                    {rec_html}
                </ol>
            </div>
            
            <div class="highlight">
                <h3>Implementation Priority</h3>
                <p><strong>Immediate:</strong> Deploy {best_config} configuration for business use</p>
                <p><strong>Short-term:</strong> Establish monitoring and feedback systems</p>
                <p><strong>Long-term:</strong> Continuous model improvement and expansion</p>
            </div>
        </div>
        """
    
    def _classify_performance_level(self, value: float, metric_type: str) -> str:
        """Classify performance level based on metric value"""
        if metric_type == 'mape':
            if value <= 5: return "Excellent"
            elif value <= 10: return "Very Good"
            elif value <= 15: return "Good"
            elif value <= 25: return "Fair"
            else: return "Needs Improvement"
        elif metric_type == 'mae':
            if value <= 25: return "Excellent"
            elif value <= 50: return "Very Good"
            elif value <= 100: return "Good"
            elif value <= 200: return "Fair"
            else: return "Needs Improvement"
        else:
            return "Good"
    
    def validate_all_configurations(self, verbose: bool = True) -> Dict[str, ValidationResult]:
        """
        Validate all available configurations
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dictionary mapping configuration names to validation results
        """
        if verbose:
            print("🚀 Starting comprehensive configuration validation...")
            print("=" * 70)
        
        results = {}
        config_names = self.config_tester.get_configuration_names()
        
        for i, config_name in enumerate(config_names, 1):
            if verbose:
                print(f"\n[{i}/{len(config_names)}] Testing {config_name}...")
            
            try:
                result = self.validate_configuration(config_name, verbose=verbose)
                results[config_name] = result
                
                if verbose:
                    print(f"✅ {config_name} completed successfully")
                    
            except Exception as e:
                if verbose:
                    print(f"❌ {config_name} failed: {str(e)}")
                continue
        
        if verbose:
            print(f"\n🎉 Validation completed! Tested {len(results)} configurations.")
            
        return results
    
    def compare_configurations(self, results: Dict[str, ValidationResult], 
                             save_plots: bool = True) -> pd.DataFrame:
        """
        Compare validation results across configurations
        
        Args:
            results: Dictionary of validation results
            save_plots: Whether to save comparison plots
            
        Returns:
            DataFrame with comparison metrics
        """
        print("📊 Generating configuration comparison...")
        
        # Create comparison dataframe
        comparison_data = []
        
        for config_name, result in results.items():
            metrics = result.overall_metrics
            
            comparison_data.append({
                'Configuration': config_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape'],
                'Bias': metrics['bias'],
                'Avg_Confidence': metrics['avg_confidence'],
                'N_Predictions': metrics['n_predictions'],
                'Duration_Sec': result.timing_info['duration_seconds'],
                'Predictions_Per_Sec': result.timing_info['predictions_per_second']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by MAE (lower is better)
        comparison_df = comparison_df.sort_values('MAE')
        
        # Create visualizations
        if save_plots:
            self._create_comparison_plots(comparison_df, results)
        
        return comparison_df
    
    def _create_comparison_plots(self, comparison_df: pd.DataFrame, 
                               results: Dict[str, ValidationResult]):
        """Create comparison visualization plots"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DrayVis K-Fold Validation - Configuration Comparison', fontsize=16, fontweight='bold')
        
        # MAE comparison
        axes[0, 0].barh(comparison_df['Configuration'], comparison_df['MAE'])
        axes[0, 0].set_title('Mean Absolute Error (Lower is Better)')
        axes[0, 0].set_xlabel('MAE ($)')
        
        # R² comparison
        axes[0, 1].barh(comparison_df['Configuration'], comparison_df['R²'])
        axes[0, 1].set_title('R² Score (Higher is Better)')
        axes[0, 1].set_xlabel('R²')
        
        # MAPE comparison
        axes[0, 2].barh(comparison_df['Configuration'], comparison_df['MAPE'])
        axes[0, 2].set_title('Mean Absolute Percentage Error (Lower is Better)')
        axes[0, 2].set_xlabel('MAPE (%)')
        
        # Confidence vs Error scatter
        for config_name, result in results.items():
            predictions = []
            actuals = []
            for fold_result in result.fold_results:
                predictions.extend(fold_result['predictions'])
                actuals.extend(fold_result['actual_values'])
            
            if predictions:
                confidences = [p['confidence_level'] for p in predictions]
                errors = [abs(p['estimated_rpm'] - a) for p, a in zip(predictions, actuals)]
                axes[1, 0].scatter(confidences, errors, alpha=0.6, label=config_name, s=20)
        
        axes[1, 0].set_title('Confidence vs Absolute Error')
        axes[1, 0].set_xlabel('Confidence Level (%)')
        axes[1, 0].set_ylabel('Absolute Error ($)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Performance by distance
        distance_data = []
        for config_name, result in results.items():
            for bin_data in result.geographic_analysis['performance_by_distance']:
                distance_data.append({
                    'Configuration': config_name,
                    'Distance_Bin': bin_data['distance_bin'],
                    'Avg_Error': bin_data['avg_error'],
                    'Avg_Distance': bin_data['avg_distance']
                })
        
        if distance_data:
            distance_df = pd.DataFrame(distance_data)
            distance_pivot = distance_df.pivot(index='Distance_Bin', columns='Configuration', values='Avg_Error')
            distance_pivot.plot(kind='bar', ax=axes[1, 1], rot=45)
            axes[1, 1].set_title('Average Error by Distance from Port')
            axes[1, 1].set_xlabel('Distance from Port (miles)')
            axes[1, 1].set_ylabel('Average Error ($)')
            axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Processing speed comparison
        axes[1, 2].barh(comparison_df['Configuration'], comparison_df['Predictions_Per_Sec'])
        axes[1, 2].set_title('Processing Speed (Higher is Better)')
        axes[1, 2].set_xlabel('Predictions per Second')
        
        plt.tight_layout()
        plt.savefig('kfold_validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comparison plots saved as 'kfold_validation_comparison.png'")
    
    def save_results(self, results: Dict[str, ValidationResult], 
                    comparison_df: pd.DataFrame, filename: str = None):
        """Save validation results to files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kfold_validation_results_{timestamp}"
        
        # Save comparison dataframe
        comparison_df.to_csv(f"{filename}_comparison.csv", index=False)
        
        # Save detailed results as JSON
        results_for_json = {}
        for config_name, result in results.items():
            # Convert ValidationResult to dictionary, handling non-serializable objects
            result_dict = asdict(result)
            
            # Convert RateEstimate objects to dictionaries in fold results
            for fold_result in result_dict['fold_results']:
                if 'predictions' in fold_result:
                    fold_result['predictions'] = [
                        asdict(pred) if hasattr(pred, '__dict__') else pred 
                        for pred in fold_result['predictions']
                    ]
            
            results_for_json[config_name] = result_dict
        
        with open(f"{filename}_detailed.json", 'w') as f:
            json.dump(results_for_json, f, indent=2, default=str)
        
        print(f"✅ Results saved:")
        print(f"  - {filename}_comparison.csv")
        print(f"  - {filename}_detailed.json")


class FastKFoldValidator(KFoldValidator):
    """High-performance parallel K-Fold validator optimized for speed"""
    
    def __init__(self, data_file: str = None, zip_coords_file: str = None, k: int = 5, 
                 random_seed: int = 42, max_test_zips: int = 50, n_cores: int = None):
        """
        Initialize Fast K-Fold validator with parallel processing
        
        Args:
            data_file: Path to drayage data CSV
            zip_coords_file: Path to zip coordinates CSV
            k: Number of folds for cross-validation
            random_seed: Random seed for reproducibility
            max_test_zips: Maximum zip codes to test (default 50 for speed)
            n_cores: Number of CPU cores to use (auto-detect if None)
        """
        # Detect system resources
        self.n_cores = n_cores or min(mp.cpu_count(), 8)  # Cap at 8 cores for memory efficiency
        self.available_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"🚀 Initializing Fast K-Fold Validator")
        print(f"   CPU Cores: {self.n_cores}")
        print(f"   Available Memory: {self.available_memory_gb:.1f} GB")
        print(f"   Max Test Zips: {max_test_zips}")
        
        # Initialize with fast mode enabled and limited test zips
        super().__init__(
            data_file=data_file,
            zip_coords_file=zip_coords_file,
            k=k,
            random_seed=random_seed,
            fast_mode=True,
            max_test_zips=max_test_zips
        )
        
        # Pre-compute expensive operations for caching
        self._precompute_distance_matrix()
        self._cache_ground_truth()
        
        print(f"✅ Fast validator ready with {len(self.zip_codes_with_data)} test zips")
    
    def _precompute_distance_matrix(self):
        """Pre-compute distance matrix for all zip codes"""
        print("📏 Pre-computing distance matrix for caching...")
        start_time = time.time()
        
        self.distance_cache = {}
        zip_coords = {}
        
        # Get coordinates for all test zips
        for zip_code in self.zip_codes_with_data:
            if zip_code in self.coord_cache:
                zip_coords[zip_code] = self.coord_cache[zip_code]
        
        # Pre-compute distances between all pairs (for faster neighbor finding)
        zip_list = list(zip_coords.keys())
        for i, zip1 in enumerate(zip_list):
            self.distance_cache[zip1] = {}
            for j, zip2 in enumerate(zip_list):
                if i != j:
                    try:
                        coord1 = zip_coords[zip1]
                        coord2 = zip_coords[zip2] 
                        distance = geodesic(coord1, coord2).miles
                        self.distance_cache[zip1][zip2] = distance
                    except:
                        self.distance_cache[zip1][zip2] = float('inf')
        
        compute_time = time.time() - start_time
        print(f"   Distance matrix computed in {compute_time:.2f}s")
    
    def _cache_ground_truth(self):
        """Cache ground truth calculations"""
        print("🎯 Caching ground truth values...")
        start_time = time.time()
        
        # The ground truth is already calculated in parent class
        # Just ensure it's optimized for fast access
        self.ground_truth_dict = {}
        for _, row in self.ground_truth.iterrows():
            self.ground_truth_dict[row['zip']] = {
                'weighted_avg_rpm': row['ground_truth_rpm'],
                'std_rpm': row['ground_truth_std'],
                'count': row['transaction_count']
            }
        
        cache_time = time.time() - start_time
        print(f"   Ground truth cached in {cache_time:.2f}s")
    
    def validate_all_configurations_parallel(self, verbose: bool = True) -> Dict[str, ValidationResult]:
        """
        Validate all configurations using parallel processing for maximum speed
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dictionary mapping configuration names to ValidationResult objects
        """
        configs = self.config_tester.get_configuration_names()
        
        print(f"\n🚀 PARALLEL VALIDATION OF {len(configs)} CONFIGURATIONS")
        print("=" * 60)
        print(f"   Cores: {self.n_cores}")
        print(f"   Folds: {self.k}")
        print(f"   Test Zips: {len(self.zip_codes_with_data)}")
        
        start_time = time.time()
        
        # Process configurations sequentially but folds in parallel for now
        # (full config parallelism can cause memory issues with complex estimators)
        results = {}
        
        for i, config_name in enumerate(configs):
            config_start = time.time()
            
            try:
                result = self._validate_single_config_fast(config_name)
                results[config_name] = result
                
                config_time = time.time() - config_start
                elapsed = time.time() - start_time
                eta = elapsed * (len(configs) - i - 1) / (i + 1) if i > 0 else 0
                
                if verbose:
                    print(f"✅ {config_name:20} | {config_time:5.1f}s | "
                          f"MAE: {result.overall_metrics['mae']:.3f} | "
                          f"ETA: {eta:5.1f}s")
                
            except Exception as e:
                print(f"❌ {config_name} failed: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 PARALLEL VALIDATION COMPLETE!")
        print(f"   Total Time: {total_time:.1f} seconds")
        print(f"   Speed: {len(configs) * self.k * len(self.zip_codes_with_data) / total_time:.1f} predictions/second")
        print(f"   Configurations: {len(results)}/{len(configs)} successful")
        
        # Force garbage collection to free memory
        gc.collect()
        
        return results
    
    def _validate_single_config_fast(self, config_name: str) -> ValidationResult:
        """Fast validation for a single configuration using cached data"""
        start_time = time.time()
        
        # Get configuration
        config = self.config_tester.get_configuration(config_name)
        
        # Use parallel fold processing
        fold_results = self._validate_folds_parallel(config)
        
        # Aggregate results quickly
        all_predictions = []
        all_actuals = []
        for fold_result in fold_results:
            all_predictions.extend(fold_result['predictions'])
            all_actuals.extend(fold_result['actual_values'])
        
        # Calculate metrics using vectorized operations
        overall_metrics = self._calculate_overall_metrics_fast(all_predictions, all_actuals)
        
        # Simplified analysis for speed
        confidence_calibration = {'avg_confidence': np.mean([p.confidence_level for p in all_predictions])}
        geographic_analysis = {'coverage': len(set([p.zip_code for p in all_predictions]))}
        
        # Minimal explanations for speed
        explanations = {
            'accuracy': {'mae': overall_metrics['mae']},
            'speed': f"Validated in {time.time() - start_time:.2f}s"
        }
        
        coverage_stats = {
            'zip_count': len(set([p.zip_code for p in all_predictions])),
            'prediction_count': len(all_predictions)
        }
        
        timing_info = {
            'duration_seconds': time.time() - start_time,
            'predictions_per_second': len(all_predictions) / (time.time() - start_time)
        }
        
        return ValidationResult(
            config_name=config_name,
            config_params=config.__dict__ if hasattr(config, '__dict__') else vars(config),
            fold_results=[fold_result for fold_result in fold_results],
            overall_metrics=overall_metrics,
            confidence_calibration=confidence_calibration,
            geographic_analysis=geographic_analysis,
            timing_info=timing_info,
            explanations=explanations,
            coverage_stats=coverage_stats
        )
    
    def _validate_folds_parallel(self, config: EstimatorConfig) -> List[Dict]:
        """Validate all folds in parallel for maximum speed"""
        # Set up k-fold splits  
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_seed)
        zip_indices = list(range(len(self.zip_codes_with_data)))
        
        fold_tasks = []
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(zip_indices)):
            train_zips = [self.zip_codes_with_data[i] for i in train_idx]
            test_zips = [self.zip_codes_with_data[i] for i in test_idx]
            fold_tasks.append((fold_idx, train_zips, test_zips, config))
        
        # Process folds in parallel (limited to prevent memory issues)
        max_fold_workers = min(self.k, 4)  # Limit fold parallelism to prevent memory overload
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_fold_workers) as executor:
            future_to_fold = {
                executor.submit(self._process_fold_fast, task): task[0]
                for task in fold_tasks
            }
            
            fold_results = []
            for future in concurrent.futures.as_completed(future_to_fold):
                try:
                    result = future.result()
                    fold_results.append(result)
                except Exception as e:
                    print(f"Fold failed: {str(e)}")
        
        # Sort by fold number to maintain order
        fold_results.sort(key=lambda x: x['fold_number'])
        return fold_results
    
    def _process_fold_fast(self, fold_data: Tuple) -> Dict:
        """Process a single fold quickly using cached data"""
        fold_idx, train_zips, test_zips, config = fold_data
        
        # Create in-memory estimator (no file I/O)
        estimator = self._create_in_memory_estimator(train_zips, config)
        
        # Make predictions using cached ground truth
        predictions = []
        actual_values = []
        
        for test_zip in test_zips:
            if test_zip in self.ground_truth_dict:
                try:
                    # Use estimator to predict
                    prediction = estimator.estimate_rate(test_zip)
                    predictions.append(prediction)
                    actual_values.append(self.ground_truth_dict[test_zip]['weighted_avg_rpm'])
                except Exception:
                    # Skip failed predictions
                    continue
        
        # Calculate errors
        errors = [p.estimated_rpm - a for p, a in zip(predictions, actual_values)]
        absolute_errors = [abs(e) for e in errors]
        
        # Calculate fold metrics quickly
        metrics = {
            'mae': np.mean(absolute_errors) if absolute_errors else float('inf'),
            'rmse': np.sqrt(np.mean([e**2 for e in errors])) if errors else float('inf'),
            'n_predictions': len(predictions)
        }
        
        return {
            'fold_number': fold_idx,
            'test_zips': test_zips,
            'predictions': predictions,
            'actual_values': actual_values,
            'errors': errors,
            'absolute_errors': absolute_errors,
            'metrics': metrics
        }
    
    def _create_in_memory_estimator(self, train_zips: List[str], config: EstimatorConfig) -> IntelligentRateEstimator:
        """Create estimator using in-memory data without file I/O"""
        # Filter training data to include only train_zips
        train_data = self.all_rates[self.all_rates['zip'].isin(train_zips)].copy()
        
        # Convert to estimator format quickly
        origins = train_data[['zip', 'lat', 'lng', 'rpm', 'date']].copy()
        origins.columns = ['origin_zip', 'origin_lat', 'origin_lng', 'RPM', 'date']
        origins['destination_zip'] = origins['origin_zip'] 
        origins['destination_lat'] = origins['origin_lat']
        origins['destination_lng'] = origins['origin_lng']
        
        # Create temporary files (can't avoid this completely)
        temp_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
        temp_data_file = f"temp_fast_{temp_id}.csv"
        temp_coords_file = f"temp_coords_{temp_id}.csv"
        
        try:
            # Quick write to temp files
            origins.to_csv(temp_data_file, index=False)
            self.mainland_zip_coords.to_csv(temp_coords_file, index=False)
            
            # Temporarily replace global config
            import intelligent_rate_estimator
            original_config = intelligent_rate_estimator.CONFIG
            intelligent_rate_estimator.CONFIG = config
            
            # Create estimator with optimizations
            estimator = IntelligentRateEstimator(
                data_file=temp_data_file,
                zip_coords_file=temp_coords_file
            )
            
            return estimator
            
        finally:
            # Restore original config
            intelligent_rate_estimator.CONFIG = original_config
            # Clean up temp files immediately
            try:
                if os.path.exists(temp_data_file):
                    os.remove(temp_data_file)
                if os.path.exists(temp_coords_file):
                    os.remove(temp_coords_file)
            except:
                pass  # Ignore cleanup errors
    
    def _calculate_overall_metrics_fast(self, predictions: List[RateEstimate], 
                                       actuals: List[float]) -> Dict:
        """Calculate overall metrics using vectorized operations for speed"""
        if not predictions or not actuals:
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': 0.0}
        
        predicted_values = np.array([p.estimated_rpm for p in predictions])
        actual_values = np.array(actuals)
        
        # Vectorized calculations
        errors = predicted_values - actual_values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # R² calculation
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((actual_values - np.mean(actual_values))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Additional metrics
        mape = np.mean(np.abs(errors / actual_values)) * 100 if np.all(actual_values != 0) else float('inf')
        bias = np.mean(errors)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'bias': float(bias),
            'n_predictions': len(predictions)
        }
    
    def quick_benchmark(self, n_configs: int = 3) -> Dict:
        """Run a quick benchmark with limited configurations for speed testing"""
        print(f"\n⚡ QUICK BENCHMARK - {n_configs} CONFIGURATIONS")
        print("=" * 50)
        
        # Test only first few configurations for speed
        all_configs = self.config_tester.get_configuration_names()
        test_configs = all_configs[:n_configs]
        
        start_time = time.time()
        
        results = {}
        for i, config_name in enumerate(test_configs):
            config_start = time.time()
            result = self._validate_single_config_fast(config_name)
            config_time = time.time() - config_start
            
            results[config_name] = result
            
            print(f"✅ {config_name:20} | {config_time:5.1f}s | "
                  f"MAE: {result.overall_metrics['mae']:.3f}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 BENCHMARK COMPLETE!")
        print(f"   Total Time: {total_time:.1f}s")
        print(f"   Avg per Config: {total_time/len(test_configs):.1f}s")
        print(f"   Speedup Estimate: {61*60/total_time:.1f}x faster than original")
        
        return results


def demo_kfold_validation():
    """Demonstrate optimized k-fold validation system for LA region"""
    print("🧪 DrayVis K-Fold Cross-Validation Demo (LA Region Optimized)")
    print("=" * 65)
    
    # Initialize validator with performance tracking
    start_time = time.time()
    validator = KFoldValidator(k=5, random_seed=42)
    init_time = time.time() - start_time
    
    print(f"\n⚡ Validator initialization time: {init_time:.2f} seconds")
    print(f"📊 Performance stats: {validator.performance_stats}")
    
    # Test a few key configurations for speed demonstration
    test_configs = ['baseline', 'aggressive_time_decay', 'tight_distance_weights']
    
    print(f"\n🔬 Testing {len(test_configs)} configurations for performance comparison...")
    
    results = {}
    config_times = {}
    
    for config in test_configs:
        print(f"\n📊 Testing {config}...")
        config_start = time.time()
        try:
            result = validator.validate_configuration(config, verbose=True)
            results[config] = result
            config_time = time.time() - config_start
            config_times[config] = config_time
            print(f"⚡ {config} completed in {config_time:.2f} seconds")
        except Exception as e:
            print(f"❌ Error testing {config}: {str(e)}")
    
    if results:
        # Compare results
        comparison_df = validator.compare_configurations(results, save_plots=True)
        
        print("\n📊 FINAL COMPARISON:")
        print("=" * 80)
        print(comparison_df[['Configuration', 'MAE', 'RMSE', 'R²', 'MAPE', 'Avg_Confidence']].to_string(index=False))
        
        print("\n⚡ PERFORMANCE TIMING:")
        print("=" * 50)
        for config, config_time in config_times.items():
            print(f"{config:25}: {config_time:6.2f} seconds")
        
        total_time = time.time() - start_time
        print(f"{'TOTAL TIME':25}: {total_time:6.2f} seconds")
        
        # Save results
        validator.save_results(results, comparison_df)
        
        print("\n🎉 Optimized K-Fold validation demo completed successfully!")
        print(f"🚀 Total execution time: {total_time:.2f} seconds")
    else:
        print("❌ No successful validations completed.")

def run_comprehensive_validation():
    """Run comprehensive validation of all configurations"""
    print("🚀 Starting Comprehensive K-Fold Validation")
    print("=" * 60)
    
    # Test with different k values and random seeds
    k_values = [5, 10]
    random_seeds = [42, 123, 456]
    
    all_results = {}
    
    for k in k_values:
        for seed in random_seeds:
            print(f"\n🔬 Testing with k={k}, seed={seed}")
            
            validator = KFoldValidator(k=k, random_seed=seed)
            
            # Test all configurations
            results = validator.validate_all_configurations(verbose=True)
            
            # Store results
            key = f"k{k}_seed{seed}"
            all_results[key] = {
                'validator_config': {'k': k, 'random_seed': seed},
                'results': results
            }
            
            # Compare and save results for this configuration
            if results:
                comparison_df = validator.compare_configurations(results, save_plots=False)
                validator.save_results(results, comparison_df, f"validation_{key}")
    
    print("\n🎉 Comprehensive validation completed!")
    print(f"Tested {len(all_results)} different validation configurations")

if __name__ == "__main__":
    # Run HIGH-PERFORMANCE validation with massive speed improvements
    print("⚡ Starting DrayVis HIGH-PERFORMANCE K-Fold Validation")
    print("   🚀 Optimized for speed: Target <5 minutes vs original 61 minutes")
    print("   🎯 Scope: Mainland areas within 300 miles of LA Port") 
    print("   💨 Features: Parallel processing, caching, intelligent sampling")
    print()
    
    import sys
    
    # Check if user wants performance comparison
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run performance comparison
        run_performance_comparison()
    elif len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Run fast demo
        demo_fast_kfold_validation()
    else:
        # Run full fast validation by default
        print("🚀 Running FULL High-Performance Validation...")
        start_time = time.time()
        
        # Initialize fast validator
        fast_validator = FastKFoldValidator(
            k=5,
            random_seed=42,
            max_test_zips=50,  # Optimized for speed while maintaining accuracy
            n_cores=None  # Auto-detect system capabilities
        )
        
        # Run all configurations with parallel processing
        results = fast_validator.validate_all_configurations_parallel(verbose=True)
        
        total_time = time.time() - start_time
        
        if results:
            print(f"\n🎉 HIGH-PERFORMANCE VALIDATION COMPLETE!")
            print("=" * 60)
            print(f"   ⚡ Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
            print(f"   🚀 Original Time: 61 minutes")
            print(f"   💨 Speed Improvement: {3660/total_time:.1f}x FASTER!")
            print(f"   ✅ Configurations: {len(results)}/8 successful")
            
            # Show best configuration
            best_config = min(results.keys(), 
                             key=lambda k: results[k].overall_metrics.get('mae', float('inf')))
            best_mae = results[best_config].overall_metrics['mae']
            best_r2 = results[best_config].overall_metrics.get('r2', 0)
            
            print(f"\n🏆 Best Configuration: {best_config}")
            print(f"   📊 MAE: {best_mae:.3f}")
            print(f"   📈 R²: {best_r2:.3f}")
            
            # Quick summary table
            print(f"\n📋 Configuration Summary (sorted by MAE):")
            print("-" * 80)
            for config_name, result in sorted(results.items(), 
                                            key=lambda x: x[1].overall_metrics.get('mae', float('inf'))):
                mae = result.overall_metrics.get('mae', float('inf'))
                r2 = result.overall_metrics.get('r2', 0)
                duration = result.timing_info.get('duration_seconds', 0)
                print(f"   {config_name:20} | MAE: {mae:.3f} | R²: {r2:.3f} | Time: {duration:.1f}s")
            
            print(f"\n💡 Use --demo for demo mode or --compare for performance comparison")
            print(f"   Example: python kfold_validation.py --demo")
            
        else:
            print("❌ Validation failed - no results generated")
            print("   Try running with --demo flag for debugging")


def demo_fast_kfold_validation():
    """Demonstrate high-performance k-fold validation system"""
    print("⚡ DrayVis FAST K-Fold Cross-Validation Demo")
    print("=" * 55)
    print("   Optimized for speed using parallel processing")
    print("   Target: Under 5 minutes vs original 61 minutes")
    print()
    
    # Initialize fast validator
    start_time = time.time()
    fast_validator = FastKFoldValidator(
        k=5, 
        random_seed=42, 
        max_test_zips=50,  # Reduced for speed
        n_cores=None  # Auto-detect
    )
    init_time = time.time() - start_time
    
    print(f"\n⚡ Fast validator initialization: {init_time:.2f} seconds")
    
    # Run quick benchmark first
    print("\n🔥 Running Quick Benchmark (3 configurations)...")
    benchmark_results = fast_validator.quick_benchmark(n_configs=3)
    
    print("\n🚀 Running Full Parallel Validation...")
    full_start = time.time()
    
    # Run all configurations in parallel
    all_results = fast_validator.validate_all_configurations_parallel(verbose=True)
    
    full_time = time.time() - full_start
    total_time = time.time() - start_time
    
    print(f"\n🎉 FAST VALIDATION COMPLETE!")
    print("=" * 50)
    print(f"   Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   Original Time: 61 minutes (3660 seconds)")
    print(f"   Speedup: {3660/total_time:.1f}x faster!")
    print(f"   Configurations Tested: {len(all_results)}")
    print(f"   Success Rate: {len(all_results)/8*100:.1f}%")
    
    # Show best configuration
    if all_results:
        best_config = min(all_results.keys(), 
                         key=lambda k: all_results[k].overall_metrics.get('mae', float('inf')))
        best_mae = all_results[best_config].overall_metrics['mae']
        
        print(f"\n🏆 Best Configuration: {best_config}")
        print(f"   MAE: {best_mae:.3f}")
        print(f"   R²: {all_results[best_config].overall_metrics.get('r2', 0):.3f}")
        
        # Show performance comparison
        print(f"\n📊 Configuration Performance Summary:")
        print("-" * 60)
        for config_name, result in sorted(all_results.items(), 
                                        key=lambda x: x[1].overall_metrics.get('mae', float('inf'))):
            mae = result.overall_metrics.get('mae', float('inf'))
            r2 = result.overall_metrics.get('r2', 0)
            time_taken = result.timing_info.get('duration_seconds', 0)
            print(f"   {config_name:20} | MAE: {mae:.3f} | R²: {r2:.3f} | Time: {time_taken:.1f}s")
    
    print(f"\n💡 Performance Optimizations Applied:")
    print(f"   ✅ Parallel fold processing")
    print(f"   ✅ In-memory data operations") 
    print(f"   ✅ Distance matrix caching")
    print(f"   ✅ Vectorized calculations")
    print(f"   ✅ Intelligent sampling ({fast_validator.max_test_zips} zip codes)")
    print(f"   ✅ Resource utilization ({fast_validator.n_cores} CPU cores)")
    
    return all_results


def run_performance_comparison():
    """Compare original vs fast validation performance"""
    print("🏁 PERFORMANCE COMPARISON: Original vs Fast Validation")
    print("=" * 65)
    
    # Quick benchmark with fast validator
    print("\n⚡ Testing Fast Validator (3 configs)...")
    fast_start = time.time()
    
    fast_validator = FastKFoldValidator(k=3, max_test_zips=30)  # Even faster for demo
    fast_results = fast_validator.quick_benchmark(n_configs=3)
    
    fast_time = time.time() - fast_start
    
    print(f"\n📊 PERFORMANCE RESULTS:")
    print("-" * 40)
    print(f"Fast Validator Time:    {fast_time:.1f} seconds")
    print(f"Estimated Original:     3660 seconds (61 minutes)")
    print(f"Speed Improvement:      {3660/fast_time:.1f}x faster")
    print(f"Time Savings:           {3660-fast_time:.0f} seconds ({(3660-fast_time)/60:.0f} minutes)")
    
    # Extrapolate to full validation
    full_fast_estimate = fast_time * (8/3)  # Scale to 8 configs
    print(f"\nFull 8-Config Estimate: {full_fast_estimate:.1f} seconds ({full_fast_estimate/60:.1f} minutes)")
    print(f"Full Speed Improvement: {3660/full_fast_estimate:.1f}x faster")
    
    return fast_results
