#!/usr/bin/env python3
"""
Port Drayage Dummy Data Generator

Generates realistic dummy data for port drayage operations including:
- 13 curated port zip codes
- 50 real drayage carriers with pricing multipliers
- Distance-based rate structure
- Import/Export order types
- Geographic coordinate calculation
"""

import csv
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd

class PortDrayageDummyGenerator:
    def __init__(self):
        # Port zip code (special - origin for imports, destination for exports)
        self.port_zip = '90802'
        
        # Curated LA area zip codes up to 200 miles from port, organized by distance bucket
        self.zip_buckets = {
            '0-25': [
                '91746', '90032', '90007', '92661', '90241', '90211', '90079', '90039',
                '92660', '90029', '91803', '92868', '90744', '90024', '90813', '90603',
                '90010', '90008', '90025', '90071', '90090', '90715', '90822', '92869',
                '90755', '90064', '92647', '91755', '92625', '92840', '90602', '90073',
                '90623', '90810', '90242', '90002', '90045', '92617', '92832', '90280',
                '92683', '90044', '92807', '90278', '92844', '92821', '90403', '90026',
                '90301', '90291'
            ],
            '25-50': [
                '92678', '91361', '90290', '91765', '92602', '91107', '91702', '91016',
                '92620', '91401', '92808', '92692', '91606', '92653', '91316', '91008',
                '93510', '91342', '91387', '92883', '91343', '91046', '91706', '91371',
                '91764', '91607', '91214', '91608', '91604', '91326', '91362', '91010',
                '91302', '92530', '91350', '92880', '91766', '92675', '91364', '91335',
                '91501', '91001', '92688', '90263', '91411', '91722', '92629', '92501',
                '91103', '92672', '91768', '91377', '91759', '91605', '91210', '91790',
                '92505', '91351', '91321', '91761', '91436', '92503', '92624', '91202',
                '92879', '93063', '91701', '92673', '92335', '91750', '91739', '92679',
                '91330', '92336', '91324', '91306', '91104', '91741', '91356', '91789',
                '92024', '93012', '92007', '93535', '92054', '92059', '92571', '92069'
            ],
            '50-75': [
                '93035', '92590', '92394', '92562', '92009', '92378', '92377', '93060',
                '92404', '92518', '92322', '92545', '91354', '93532', '93001', '93544',
                '93563', '92532', '92325', '92583', '92371', '93004', '92345', '92395',
                '93551', '92401', '91320', '92324', '92411', '92081', '92582', '92028',
                '92346', '92570', '92223', '93543', '92376', '92586', '92543', '93033',
                '92584', '92358', '92408', '91355', '92591', '92374', '92352', '92055',
                '91384', '93021', '92397', '92344', '92029', '93561', '93108', '92234',
                '92025', '93225', '92268', '93243', '92065', '93252', '92549', '92220'
            ],
            '75-100': [
                '92305', '92082', '92110', '92264', '92347', '92134', '92104', '92282',
                '92102', '92117', '92130', '92544', '92126', '92140', '92262', '92145',
                '92064', '92108', '92105', '93516', '92121', '91945', '93109', '92106',
                '92342', '91941', '92027', '92021', '92307', '92339', '91977', '92120',
                '92256', '92109', '92356', '92067', '92240', '93067', '92236', '91980',
                '93460', '92260', '93276', '92253', '93301', '92211', '91948', '91915'
            ],
            '100-125': [
                '93311', '93110', '93558', '92173', '93306', '93313', '92252', '93220',
                '91911', '93111', '93105', '93268', '93309', '93254', '92210', '92154',
                '93117', '91978', '93312', '92327', '92276', '92241', '91917', '93304',
                '91935', '92019', '91963', '91916', '92274', '91910', '93436', '93555',
                '92275', '92004', '91905', '92259', '93437', '93287', '93427', '92277'
            ],
            '125-150': [
                '93441', '93224', '93240', '93440', '93463', '93207', '91906', '93314',
                '93280', '93454', '93255', '92278', '93226', '93250', '93308', '93562',
                '92254', '93283', '93285', '93206', '92281', '93201', '93238', '93208',
                '92239', '93449', '93434', '93453', '93249', '93444'
            ],
            '150-175': [
                '93433', '93420', '93218', '93401', '93270', '93424', '92249', '92231',
                '93272', '92233', '93267', '93258', '93239', '93256', '92257', '93292',
                '92389', '93522', '93244', '92266', '92364', '93402', '93446', '93271',
                '93245', '92384', '93274', '93277', '93221'
            ],
            '175-200': [
                '93615', '93202', '93230', '92332', '93422', '93442'
            ]
        }
        
        # 50 real drayage carriers with ultra-smooth multiplier distribution
        # Multipliers range from 0.95 to 1.05 in tiny increments for maximum smoothness
        # Volume weights: 0-15 scale where larger carriers get higher weights (more frequent selection)
        self.carriers = {
            'Hub Group': {'multiplier': 0.950, 'weight': 15},  # Largest carrier
            'J.B. Hunt Transport': {'multiplier': 0.952, 'weight': 15},
            'Schneider National': {'multiplier': 0.954, 'weight': 15},
            'XPO Logistics': {'multiplier': 0.956, 'weight': 14},
            'C.H. Robinson': {'multiplier': 0.958, 'weight': 13},
            'Amazon Trucking': {'multiplier': 0.960, 'weight': 10},
            'Dependable Highway Express': {'multiplier': 0.962, 'weight': 8},
            'MTK Transportation': {'multiplier': 0.964, 'weight': 7},
            'United Logistic Services Group': {'multiplier': 0.966, 'weight': 5},
            'Pac Anchor Transportation': {'multiplier': 0.968, 'weight': 5},
            'Ecology Auto Parts': {'multiplier': 0.970, 'weight': 5},
            'On Time Truckers': {'multiplier': 0.972, 'weight': 5},
            'Precision Worldwide Logistics': {'multiplier': 0.974, 'weight': 4},
            'Gill Roadway Inc': {'multiplier': 0.976, 'weight': 4},
            'Freight Horse Express': {'multiplier': 0.978, 'weight': 4},
            'Total Distribution Service': {'multiplier': 0.980, 'weight': 3},
            'JYC Trucking LLC': {'multiplier': 0.982, 'weight': 2},
            'QX Logistix LLC': {'multiplier': 0.984, 'weight': 2},
            '7 Star Logistics': {'multiplier': 0.986, 'weight': 2},
            'Year-Round Enterprises': {'multiplier': 0.988, 'weight': 2},
            'Hight Logistics': {'multiplier': 0.990, 'weight': 2},
            'JT Freight Trucking': {'multiplier': 0.992, 'weight': 2},
            'SOPAC - Southern Pacific Logistics': {'multiplier': 0.994, 'weight': 2},
            'Golden King Transport': {'multiplier': 0.996, 'weight': 2},
            'National Road Logistics': {'multiplier': 0.998, 'weight': 2},
            'DK Express': {'multiplier': 1.000, 'weight': 2},
            'Lectro Trucking': {'multiplier': 1.002, 'weight': 2},
            'MTZ Trucking LLC': {'multiplier': 1.004, 'weight': 2},
            '360 Global Transportation': {'multiplier': 1.006, 'weight': 1},
            'Access One Transport': {'multiplier': 1.008, 'weight': 1},
            'Airborne Freight Lines': {'multiplier': 1.010, 'weight': 1},
            'All Harbor Transport LLC': {'multiplier': 1.012, 'weight': 1},
            'All Modal Transportation': {'multiplier': 1.014, 'weight': 1},
            'All Ports Logistics': {'multiplier': 1.016, 'weight': 1},
            'All Seasons Trucking LLC': {'multiplier': 1.018, 'weight': 1},
            'Alliance Specialized': {'multiplier': 1.020, 'weight': 1},
            'Ape Express Trucking': {'multiplier': 1.022, 'weight': 1},
            'Avila Trucking LLC': {'multiplier': 1.024, 'weight': 1},
            'Ayden Transport': {'multiplier': 1.026, 'weight': 1},
            'Basuta Brothers': {'multiplier': 1.028, 'weight': 1},
            'BNDZ Transportation LLC': {'multiplier': 1.030, 'weight': 1},
            'Cal U Transport': {'multiplier': 1.032, 'weight': 1},
            'Campos Trucking': {'multiplier': 1.034, 'weight': 1},
            'Can Trail Transportation LLC': {'multiplier': 1.036, 'weight': 1},
            'Cardona Trucking': {'multiplier': 1.038, 'weight': 1},
            'Cargo Legion': {'multiplier': 1.040, 'weight': 0},  # Smallest carriers
            'Cecia Transportation': {'multiplier': 1.042, 'weight': 0},
            'Chicas & Co Logistics': {'multiplier': 1.044, 'weight': 0},
            'CSC Logistics': {'multiplier': 1.046, 'weight': 0},
            'Drayage Group Inc': {'multiplier': 1.048, 'weight': 0}
        }
        
        # Create weighted carrier list for selection
        self.weighted_carriers = []
        for carrier, data in self.carriers.items():
            weight = data['weight'] + 1  # Add 1 so even weight=0 carriers appear once
            self.weighted_carriers.extend([carrier] * weight)
        
        # Create weighted bucket list to favor longer distances
        bucket_weights = {
            '0-25': 8,    # Lowest weight - closest to port
            '25-50': 7,
            '50-75': 6,
            '75-100': 5,
            '100-125': 4,
            '125-150': 3,
            '150-175': 2,
            '175-200': 1  # Highest weight - farthest from port
        }
        self.weighted_buckets = []
        for bucket, weight in bucket_weights.items():
            self.weighted_buckets.extend([bucket] * weight)
        
        # Load US zip coordinates
        self.zip_coords = self.load_zip_coordinates()
        
        # All zip codes we need (port + curated destinations)
        self.curated_zips = sum(self.zip_buckets.values(), [])
        
        # Filter out zip codes that are within 2 miles of the port
        self.curated_zips = self.filter_close_zips(self.curated_zips)
        
        # Update zip buckets with filtered zips
        self.update_zip_buckets_after_filtering()
        
        self.all_zips = [self.port_zip] + self.curated_zips
        
    def load_zip_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """Load zip code coordinates from CSV file."""
        coords = {}
        try:
            with open('data/us_zip_coordinates.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    zip_code = row['ZIP']
                    lat = float(row['LAT'])
                    lng = float(row['LNG'])
                    coords[zip_code] = (lat, lng)
        except FileNotFoundError:
            print("Warning: us_zip_coordinates.csv not found. Using default coordinates.")
            # Default coordinates for LA area
            coords[self.port_zip] = (33.7701, -118.1937)  # LA port coords
            for zip_code in self.curated_zips:
                coords[zip_code] = (33.7701, -118.1937)  # Default LA area coords
        return coords
    
    def filter_close_zips(self, zip_list: List[str]) -> List[str]:
        """Filter out zip codes that are within 2 miles of the port."""
        filtered_zips = []
        port_coords = self.zip_coords.get(self.port_zip, (33.745762, -118.208042))
        
        for zip_code in zip_list:
            if zip_code in self.zip_coords:
                zip_coords = self.zip_coords[zip_code]
                distance = self.haversine_distance(
                    port_coords[0], port_coords[1],
                    zip_coords[0], zip_coords[1]
                )
                # Only include zips that are more than 2 miles away from the port
                if distance > 2.0:
                    filtered_zips.append(zip_code)
                else:
                    print(f"🚫 Excluded {zip_code} - {distance:.2f} miles from port")
            else:
                # If no coordinates found, include it (better than excluding potentially valid zips)
                filtered_zips.append(zip_code)
        
        return filtered_zips
    
    def update_zip_buckets_after_filtering(self):
        """Update zip buckets after filtering out close zips."""
        for bucket_name, zip_list in self.zip_buckets.items():
            filtered_list = []
            for zip_code in zip_list:
                if zip_code in self.curated_zips:
                    filtered_list.append(zip_code)
            self.zip_buckets[bucket_name] = filtered_list
            print(f"📊 Bucket {bucket_name}: {len(filtered_list)} zips (after filtering)")
    
    def haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate the great circle distance between two points on Earth."""
        # Convert latitude and longitude to radians
        lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in miles
        r = 3956
        return c * r
    
    def get_base_rate_per_mile(self, distance: float) -> float:
        """
        Calculate distance-based rate per mile component (added to $150 base rate).
        
        This is the distance component only - total rate = $150 base + (distance * this_rate)
        Distance rate decreases exponentially with distance:
        - Very short distances (0-5 miles): $12-15/mile distance component
        - Short distances (5-20 miles): $8-12/mile distance component
        - Medium distances (20-50 miles): $5-8/mile distance component
        - Long distances (50+ miles): $4-5/mile distance component
        """
        if distance <= 0:
            return 14.00
        
        # Exponential decay function: rate = base * e^(-decay * distance) + minimum
        base_rate = 14.00      # Starting rate for very short distances
        minimum_rate = 3.50    # Floor rate for very long distances
        decay_factor = 0.045   # Controls how quickly rate decreases
        
        # Calculate exponential decay
        rate = (base_rate - minimum_rate) * math.exp(-decay_factor * distance) + minimum_rate
        
        # Add small distance penalty for very short trips (under 3 miles)
        if distance < 3:
            short_penalty = (3 - distance) * 0.5  # Up to $1.50 extra for very short trips
            rate += short_penalty
        
        # Ensure rate stays within reasonable bounds
        return max(3.50, min(16.00, rate))
    
    def get_time_based_multiplier(self, date_str: str) -> float:
        """Get time-based rate multiplier with smooth market trends and seasonal patterns."""
        # Parse the date
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Calculate days from start of period (365 days ago)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Calculate how far through the year we are (0.0 to 1.0)
        days_from_start = (date_obj - start_date).days
        progress = max(0.0, min(1.0, days_from_start / 365.0))  # Clamp to valid range
        
        # Base growth trend - smooth S-curve instead of linear
        # Uses sigmoid function for more realistic market adoption curve
        base_multiplier = 0.98
        max_growth = 0.06  # Maximum 6% growth over the year
        
        # Sigmoid function for smooth S-curve growth: f(x) = 1 / (1 + e^(-k(x-0.5)))
        # Shifted and scaled to create smooth growth from 0.98 to 1.04
        k = 6  # Controls steepness of the S-curve
        sigmoid_progress = 1 / (1 + math.exp(-k * (progress - 0.5)))
        trend_multiplier = base_multiplier + (max_growth * sigmoid_progress)
        
        # Seasonal patterns using sine waves for smooth transitions
        # Day of year (0-365) converted to radians for sine function
        day_of_year = date_obj.timetuple().tm_yday
        yearly_cycle = (day_of_year - 1) / 365.0 * 2 * math.pi  # 0 to 2π over the year
        
        # Peak season (Q4 holiday rush): Oct-Dec higher rates
        # Secondary peak (back-to-school/harvest): Aug-Sep
        # Low season: Jan-Mar (post-holiday slowdown)
        seasonal_base = 0.02 * math.sin(yearly_cycle - math.pi/3)  # ±2% seasonal variation
        
        # Add quarterly business cycles (4 cycles per year)
        quarterly_cycle = 0.01 * math.sin(4 * yearly_cycle + math.pi/4)  # ±1% quarterly variation
        
        # Add monthly micro-fluctuations for market volatility
        monthly_cycle = 0.005 * math.sin(12 * yearly_cycle)  # ±0.5% monthly variation
        
        # Combine all components for smooth, realistic market trends
        final_multiplier = trend_multiplier + seasonal_base + quarterly_cycle + monthly_cycle
        
        # Ensure multiplier stays within reasonable bounds (0.95 to 1.08)
        return max(0.95, min(1.08, final_multiplier))
    
    def calculate_rate(self, origin_zip: str, dest_zip: str, carrier: str, date_str: str) -> Tuple[float, float, float]:
        """Calculate the drayage rate for a given route and carrier with time-based adjustments."""
        # Get coordinates
        if origin_zip not in self.zip_coords or dest_zip not in self.zip_coords:
            return 0.0, 0.0, 0.0
        
        origin_coords = self.zip_coords[origin_zip]
        dest_coords = self.zip_coords[dest_zip]
        
        # Calculate distance
        distance = self.haversine_distance(
            origin_coords[0], origin_coords[1],
            dest_coords[0], dest_coords[1]
        )
        
        # Get base rate per mile
        base_rpm = self.get_base_rate_per_mile(distance)
        
        # Apply carrier multiplier
        carrier_multiplier = self.carriers[carrier]['multiplier']
        
        # Apply time-based multiplier
        time_multiplier = self.get_time_based_multiplier(date_str)
        
        # Calculate final rate per mile with reduced variation
        # Add small random variation (±2% instead of larger variation)
        random_variation = random.uniform(0.99, 1.01)  # Reduced from wider range
        actual_rpm = base_rpm * carrier_multiplier * time_multiplier * random_variation
        
        # Calculate total rate with realistic minimum base rate
        # Start with $200 base rate and add distance-based pricing (higher than original $150)
        base_minimum_rate = 200.00  # Increased from $150 to ensure no low rates
        distance_rate = actual_rpm * distance
        total_rate = base_minimum_rate + distance_rate
        
        # Calculate effective RPM (total rate divided by distance)
        effective_rpm = total_rate / distance if distance > 0 else 0
        
        return distance, total_rate, effective_rpm
    
    def generate_random_date(self, start_date: datetime, end_date: datetime) -> str:
        """Generate a random date between start and end dates."""
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        random_date = start_date + timedelta(days=random_days)
        return random_date.strftime('%Y-%m-%d')
    
    def generate_records(self, num_records: int = 5000) -> List[Dict]:
        """Generate dummy drayage records with balanced distance distribution and RPM filtering."""
        records = []
        # Date range: 1yr
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        print(f"Generating {num_records} dummy drayage records...")
        
        attempts = 0
        max_attempts = num_records * 3  # Allow some buffer for RPM filtering
        
        while len(records) < num_records and attempts < max_attempts:
            if len(records) % 500 == 0 and len(records) > 0:
                print(f"Generated {len(records)} records...")
            
            attempts += 1
            
            # Weighted carrier selection for realistic market distribution
            carrier = random.choice(self.weighted_carriers)
            
            # 80% imports, 20% exports
            order_type = 'import' if random.random() < 0.80 else 'export'
            
            # Weighted bucket selection to favor longer distances
            chosen_bucket = random.choice(self.weighted_buckets)
            dest_zip = random.choice(self.zip_buckets[chosen_bucket])
            
            # Set origin/destination based on order type
            if order_type == 'import':
                origin_zip = self.port_zip
                destination_zip = dest_zip
            else:
                origin_zip = dest_zip
                destination_zip = self.port_zip
            # Generate random date
            date = self.generate_random_date(start_date, end_date)
            # Calculate rate and distance (now with time-based adjustments)
            distance, rate, rpm = self.calculate_rate(origin_zip, destination_zip, carrier, date)
            
            # Apply RPM filtering - skip records with RPM > 25
            if distance > 0 and rpm <= 25.0:
                origin_coords = self.zip_coords[origin_zip]
                dest_coords = self.zip_coords[destination_zip]
                record = {
                    'origin_zip': origin_zip,
                    'destination_zip': destination_zip,
                    'date': date,
                    'carrier': carrier,
                    'order_type': order_type,
                    'origin_lat': round(origin_coords[0], 6),
                    'origin_lng': round(origin_coords[1], 6),
                    'destination_lat': round(dest_coords[0], 6),
                    'destination_lng': round(dest_coords[1], 6),
                    'miles': round(distance, 2),
                    'rate': round(rate, 2),
                    'RPM': round(rpm, 2)
                }
                records.append(record)
        
        print(f"Generated {len(records)} valid records (filtered {attempts - len(records)} high RPM records).")
        return records
    
    def save_to_csv(self, records: List[Dict], filename: str = 'data/port_drayage_dummy_data.csv'):
        """Save records to CSV file."""
        if not records:
            print("No records to save.")
            return
        
        fieldnames = [
            'origin_zip', 'destination_zip', 'date', 'carrier', 'order_type',
            'origin_lat', 'origin_lng', 'destination_lat', 'destination_lng',
            'miles', 'rate', 'RPM'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        
        print(f"Saved {len(records)} records to {filename}")
    
    def print_summary_stats(self, records: List[Dict]):
        """Print summary statistics of the generated data."""
        if not records:
            return
        
        print("\n=== SUMMARY STATISTICS ===")
        
        # Basic counts
        print(f"Total Records: {len(records)}")
        
        # Order type distribution
        import_count = sum(1 for r in records if r['order_type'] == 'import')
        export_count = len(records) - import_count
        print(f"Import Orders: {import_count} ({import_count/len(records)*100:.1f}%)")
        print(f"Export Orders: {export_count} ({export_count/len(records)*100:.1f}%)")
        
        # Port zip distribution
        port_counts = {}
        curated_counts = {}
        for record in records:
            if record['order_type'] == 'import':
                # For imports, destination is the curated zip
                curated_zip = record['destination_zip']
                curated_counts[curated_zip] = curated_counts.get(curated_zip, 0) + 1
            else:
                # For exports, origin is the curated zip
                curated_zip = record['origin_zip']
                curated_counts[curated_zip] = curated_counts.get(curated_zip, 0) + 1
        
        print(f"\nCurated Zip Distribution:")
        for zip_code, count in sorted(curated_counts.items()):
            print(f"  {zip_code}: {count} orders")
        
        # Rate statistics
        rates = [r['rate'] for r in records]
        rpms = [r['RPM'] for r in records]
        distances = [r['miles'] for r in records]
        
        print(f"\nRate Statistics:")
        print(f"  Total Rate - Min: ${min(rates):.2f}, Max: ${max(rates):.2f}, Avg: ${sum(rates)/len(rates):.2f}")
        print(f"  RPM - Min: ${min(rpms):.2f}, Max: ${max(rpms):.2f}, Avg: ${sum(rpms)/len(rpms):.2f}")
        print(f"  Distance - Min: {min(distances):.1f}mi, Max: {max(distances):.1f}mi, Avg: {sum(distances)/len(distances):.1f}mi")
        
        # Carrier distribution (top 10)
        carrier_counts = {}
        for record in records:
            carrier = record['carrier']
            carrier_counts[carrier] = carrier_counts.get(carrier, 0) + 1
        
        print(f"\nTop 10 Carriers by Volume:")
        for carrier, count in sorted(carrier_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {carrier}: {count} orders")

def main():
    """Main function to generate dummy data."""
    generator = PortDrayageDummyGenerator()
    
    # Generate records
    records = generator.generate_records(num_records=5000)
    
    # Save to CSV
    generator.save_to_csv(records)
    
    # Print summary
    generator.print_summary_stats(records)
    
    print(f"\n✅ Port drayage dummy data generation complete!")
    print(f"📁 Data saved to: data/port_drayage_dummy_data.csv")
    print(f"🚛 {len(records)} realistic drayage records generated")
    print(f"🏭 Port: {generator.port_zip} (origin for imports, destination for exports)")
    print(f"📍 {len(generator.curated_zips)} curated destination zip codes")
    print(f"🚚 {len(generator.carriers)} real drayage carriers")

if __name__ == "__main__":
    main()
