import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedRouteRatePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        
    def load_data(self):
        """Load and examine the port drayage data"""
        self.df = pd.read_csv(self.data_path)
        print("Dataset shape:", self.df.shape)
        print("\nColumn names and types:")
        print(self.df.dtypes)
        print("\nFirst few rows:")
        print(self.df.head())
        return self.df
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calculate the bearing (compass direction) from point 1 to point 2
        Returns bearing in degrees (0-360)
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        lon1_rad = np.radians(lon1)
        lon2_rad = np.radians(lon2)
        
        # Calculate bearing
        dlon = lon2_rad - lon1_rad
        
        y = np.sin(dlon) * np.cos(lat2_rad)
        x = (np.cos(lat1_rad) * np.sin(lat2_rad) - 
             np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon))
        
        bearing_rad = np.arctan2(y, x)
        bearing_deg = np.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Radius of earth in miles
        r = 3956
        return c * r
    
    def feature_engineering(self):
        """Enhanced feature engineering with bearing and geographic features"""
        self.processed_df = self.df.copy()
        
        # Calculate bearing degree
        self.processed_df['bearing_degree'] = self.calculate_bearing(
            self.processed_df['origin_lat'],
            self.processed_df['origin_lng'],
            self.processed_df['destination_lat'],
            self.processed_df['destination_lng']
        )
        
        # Calculate cardinal direction categories
        def bearing_to_direction(bearing):
            if bearing >= 337.5 or bearing < 22.5:
                return 'N'
            elif 22.5 <= bearing < 67.5:
                return 'NE'
            elif 67.5 <= bearing < 112.5:
                return 'E'
            elif 112.5 <= bearing < 157.5:
                return 'SE'
            elif 157.5 <= bearing < 202.5:
                return 'S'
            elif 202.5 <= bearing < 247.5:
                return 'SW'
            elif 247.5 <= bearing < 292.5:
                return 'W'
            else:  # 292.5 <= bearing < 337.5
                return 'NW'
        
        self.processed_df['cardinal_direction'] = self.processed_df['bearing_degree'].apply(bearing_to_direction)
        
        # Calculate verified distance using Haversine formula
        self.processed_df['calculated_miles'] = self.calculate_haversine_distance(
            self.processed_df['origin_lat'],
            self.processed_df['origin_lng'],
            self.processed_df['destination_lat'],
            self.processed_df['destination_lng']
        )
        
        # Calculate distance difference (actual vs calculated)
        if 'miles' in self.processed_df.columns:
            self.processed_df['distance_variance'] = abs(self.processed_df['miles'] - self.processed_df['calculated_miles'])
        
        # Geographic features
        self.processed_df['lat_diff'] = abs(self.processed_df['destination_lat'] - self.processed_df['origin_lat'])
        self.processed_df['lng_diff'] = abs(self.processed_df['destination_lng'] - self.processed_df['origin_lng'])
        
        # Centroid coordinates (midpoint of route)
        self.processed_df['route_center_lat'] = (self.processed_df['origin_lat'] + self.processed_df['destination_lat']) / 2
        self.processed_df['route_center_lng'] = (self.processed_df['origin_lng'] + self.processed_df['destination_lng']) / 2
        
        # Distance-based features
        distance_col = 'miles' if 'miles' in self.processed_df.columns else 'calculated_miles'
        self.processed_df['distance_log'] = np.log1p(self.processed_df[distance_col])
        self.processed_df['distance_squared'] = self.processed_df[distance_col] ** 2
        self.processed_df['distance_sqrt'] = np.sqrt(self.processed_df[distance_col])
        
        # Rate per mile features (if rate column exists)
        rate_col = None
        for col in ['rate', 'total_rate', 'price', 'cost']:
            if col in self.processed_df.columns:
                rate_col = col
                break
        
        if rate_col and distance_col:
            self.processed_df['rate_per_mile'] = self.processed_df[rate_col] / self.processed_df[distance_col]
            self.processed_df['rate_per_mile_log'] = np.log1p(self.processed_df['rate_per_mile'])
        
        # Time-based features
        if 'date' in self.processed_df.columns:
            self.processed_df['date'] = pd.to_datetime(self.processed_df['date'])
            self.processed_df['month'] = self.processed_df['date'].dt.month
            self.processed_df['day_of_week'] = self.processed_df['date'].dt.dayofweek
            self.processed_df['quarter'] = self.processed_df['date'].dt.quarter
            self.processed_df['is_weekend'] = (self.processed_df['day_of_week'] >= 5).astype(int)
            
            # Seasonal features
            self.processed_df['season'] = self.processed_df['month'].apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                         'Spring' if x in [3, 4, 5] else
                         'Summer' if x in [6, 7, 8] else 'Fall'
            )
        
        # Encode categorical variables
        categorical_cols = ['carrier', 'order_type', 'cardinal_direction']
        if 'season' in self.processed_df.columns:
            categorical_cols.append('season')
            
        for col in categorical_cols:
            if col in self.processed_df.columns:
                le = LabelEncoder()
                self.processed_df[col + '_encoded'] = le.fit_transform(self.processed_df[col].astype(str))
                self.label_encoders[col] = le
        
        print("Enhanced feature engineering completed!")
        print(f"New features added: bearing_degree, cardinal_direction, calculated_miles, and more")
        print(f"Total columns: {len(self.processed_df.columns)}")
        
        return self.processed_df
    
    def analyze_bearing_patterns(self):
        """Analyze bearing patterns in the data"""
        if 'bearing_degree' not in self.processed_df.columns:
            print("Bearing degree not calculated. Run feature_engineering() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bearing distribution
        axes[0,0].hist(self.processed_df['bearing_degree'], bins=36, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Bearing Degrees')
        axes[0,0].set_xlabel('Bearing (degrees)')
        axes[0,0].set_ylabel('Frequency')
        
        # Cardinal direction distribution
        direction_counts = self.processed_df['cardinal_direction'].value_counts()
        axes[0,1].pie(direction_counts.values, labels=direction_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Distribution of Cardinal Directions')
        
        # Bearing vs Rate (if rate column exists)
        rate_col = None
        for col in ['rate', 'total_rate', 'price', 'cost']:
            if col in self.processed_df.columns:
                rate_col = col
                break
        
        if rate_col:
            axes[1,0].scatter(self.processed_df['bearing_degree'], self.processed_df[rate_col], alpha=0.6)
            axes[1,0].set_xlabel('Bearing Degree')
            axes[1,0].set_ylabel(f'{rate_col.title()}')
            axes[1,0].set_title(f'Bearing vs {rate_col.title()}')
            
            # Cardinal direction vs Rate
            sns.boxplot(data=self.processed_df, x='cardinal_direction', y=rate_col, ax=axes[1,1])
            axes[1,1].set_title(f'Cardinal Direction vs {rate_col.title()}')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation analysis
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.processed_df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5)
        plt.title('Enhanced Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def prepare_model_data(self, target_column=None):
        """Prepare data for machine learning models"""
        if self.processed_df is None:
            print("Data not processed. Run feature_engineering() first.")
            return None, None
        
        # Auto-detect target column if not specified
        if target_column is None:
            for col in ['rate', 'total_rate', 'price', 'cost']:
                if col in self.processed_df.columns:
                    target_column = col
                    break
        
        if target_column is None:
            print("No target column found. Please specify target_column parameter.")
            return None, None
        
        # Select feature columns
        feature_cols = []
        for col in self.processed_df.columns:
            if (col.endswith('_encoded') or 
                (self.processed_df[col].dtype in ['int64', 'float64'] and 
                 col not in [target_column, 'date'] and
                 not col.startswith('origin_') and not col.startswith('destination_'))):
                feature_cols.append(col)
        
        print(f"Target column: {target_column}")
        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        
        X = self.processed_df[feature_cols].fillna(0)
        y = self.processed_df[target_column].fillna(self.processed_df[target_column].median())
        
        return X, y, feature_cols
    
    def train_models(self, X, y, feature_cols):
        """Train multiple models and compare performance"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features for linear models
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'Enhanced Random Forest': RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if 'Linear' in name:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape,
                'actual': y_test
            }
            
            print(f"MAE: ${mae:.2f}")
            print(f"RMSE: ${rmse:.2f}")
            print(f"R²: {r2:.3f}")
            print(f"MAPE: {mape:.2f}%")
        
        self.models = results
        return results
    
    def analyze_feature_importance(self, feature_cols):
        """Analyze feature importance from Random Forest models"""
        if not self.models:
            print("No models trained. Run train_models() first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        rf_models = [(name, results) for name, results in self.models.items() 
                     if 'Random Forest' in name]
        
        for i, (name, results) in enumerate(rf_models):
            model = results['model']
            importance = model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot top 15 features
            top_features = feature_importance.head(15)
            sns.barplot(data=top_features, y='feature', x='importance', ax=axes[i])
            axes[i].set_title(f'{name} - Top 15 Feature Importance')
            axes[i].set_xlabel('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        # Return feature importance for the best Random Forest model
        best_rf = max(rf_models, key=lambda x: x[1]['r2'])
        return pd.DataFrame({
            'feature': feature_cols,
            'importance': best_rf[1]['model'].feature_importances_
        }).sort_values('importance', ascending=False)
    
    def plot_predictions(self):
        """Visualize model predictions vs actual values"""
        if not self.models:
            print("No models trained. Run train_models() first.")
            return
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for i, (name, results) in enumerate(self.models.items()):
            y_test = results['actual']
            y_pred = results['predictions']
            
            axes[i].scatter(y_test, y_pred, alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Rate ($)')
            axes[i].set_ylabel('Predicted Rate ($)')
            axes[i].set_title(f'{name}\nR² = {results["r2"]:.3f}')
            
            # Add perfect prediction line
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_route(self, route_data):
        """Predict rate for a new route"""
        if not self.models or not self.scaler:
            print("Models not trained. Run train_models() first.")
            return None
        
        # Use the best performing model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2'])
        best_model = self.models[best_model_name]['model']
        
        print(f"Using {best_model_name} for prediction...")
        
        # Create a sample prediction (you'll need to adjust this based on your input format)
        # This is a template - customize based on your actual route_data structure
        sample_data = {
            'bearing_degree': self.calculate_bearing(
                route_data.get('origin_lat', 0),
                route_data.get('origin_lng', 0),
                route_data.get('destination_lat', 0),
                route_data.get('destination_lng', 0)
            ),
            'miles': route_data.get('miles', 0),
            # Add other features as needed
        }
        
        # Convert to DataFrame and make prediction
        pred_df = pd.DataFrame([sample_data])
        prediction = best_model.predict(pred_df)[0]
        
        return {
            'predicted_rate': prediction,
            'model_used': best_model_name,
            'confidence_r2': self.models[best_model_name]['r2']
        }

# Usage example
if __name__ == "__main__":
    # Initialize the predictor
    predictor = EnhancedRouteRatePredictor('data/port_drayage_dummy_data.csv')
    
    # Load and process data
    df = predictor.load_data()
    processed_df = predictor.feature_engineering()
    
    # Analyze bearing patterns
    predictor.analyze_bearing_patterns()
    
    # Prepare model data
    X, y, feature_cols = predictor.prepare_model_data()
    
    if X is not None and y is not None:
        # Train models
        results = predictor.train_models(X, y, feature_cols)
        
        # Analyze feature importance
        feature_importance = predictor.analyze_feature_importance(feature_cols)
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Plot predictions
        predictor.plot_predictions()
        
        print("\nEnhanced Route Rate Prediction System Ready!")
        print("Key enhancements:")
        print("- Bearing degree calculations")
        print("- Cardinal direction features")
        print("- Geographic distance calculations")
        print("- Enhanced Random Forest models")
        print("- Comprehensive feature importance analysis")
