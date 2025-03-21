import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

class AirQualityDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'temperature', 'humidity', 'pm25', 'pm10',
            'no2', 'so2', 'co', 'industrial_proximity',
            'population_density', 'latitude', 'longitude'
        ]
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load the cleaned dataset"""
        try:
            self.df = pd.read_csv('finaldata.csv')
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            return self.df
        except FileNotFoundError:
            print("Warning: finaldata.csv not found. Using sample data instead.")
            return self.load_sample_data()
        
    def load_sample_data(self):
        """Generate sample data for testing and development (fallback method)"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
            'temperature': np.random.normal(25, 5, n_samples),
            'humidity': np.random.normal(60, 10, n_samples),
            'pm25': np.random.normal(50, 20, n_samples),
            'pm10': np.random.normal(75, 25, n_samples),
            'no2': np.random.normal(40, 15, n_samples),
            'so2': np.random.normal(20, 8, n_samples),
            'co': np.random.normal(2, 0.5, n_samples),
            'industrial_proximity': np.random.normal(5, 2, n_samples),
            'population_density': np.random.normal(1000, 200, n_samples),
            'latitude': np.random.normal(40, 5, n_samples),
            'longitude': np.random.normal(-74, 5, n_samples)
        }
        
        self.df = pd.DataFrame(data)
        self.df['air_quality'] = self._calculate_air_quality(self.df)
        return self.df
    
    def _calculate_air_quality(self, df):
        """Calculate air quality category based on pollutant levels"""
        conditions = [
            (df['pm25'] <= 12) & (df['pm10'] <= 54) & (df['no2'] <= 53) & 
            (df['so2'] <= 35) & (df['co'] <= 4.4),
            (df['pm25'] <= 35.4) & (df['pm10'] <= 154) & (df['no2'] <= 100) & 
            (df['so2'] <= 75) & (df['co'] <= 9.4),
            (df['pm25'] <= 55.4) & (df['pm10'] <= 254) & (df['no2'] <= 360) & 
            (df['so2'] <= 185) & (df['co'] <= 12.4),
            (df['pm25'] > 55.4) | (df['pm10'] > 254) | (df['no2'] > 360) | 
            (df['so2'] > 185) | (df['co'] > 12.4)
        ]
        choices = ['Good', 'Moderate', 'Poor', 'Hazardous']
        return np.select(conditions, choices, default='Unknown')
    
    def get_latest_data(self, n_samples=100):
        """Get the most recent n samples from the dataset"""
        return self.df.sort_values('timestamp', ascending=False).head(n_samples)
    
    def preprocess_data(self):
        """Preprocess the data for training"""
        df = self.df.copy()
        
        # Convert air quality categories to numeric values
        quality_map = {
            'Good': 0,
            'Moderate': 1,
            'Poor': 2,
            'Hazardous': 3
        }
        df['air_quality'] = df['air_quality'].map(quality_map)
        
        # Fill missing values with mean for numeric columns only
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Select features for training
        feature_columns = ['temperature', 'humidity', 'pm25', 'pm10', 'no2', 'so2', 'co']
        X = df[feature_columns]
        y = df['air_quality']
        
        return X, y
    
    def prepare_training_data(self, test_size=0.2):
        """Prepare data for model training"""
        X, y = self.preprocess_data()
        return train_test_split(X, y, test_size=test_size, random_state=42)
    
    def get_statistics(self, df=None):
        """Calculate basic statistics for the dataset"""
        if df is None:
            df = self.df
            
        stats = {
            'total_samples': len(df),
            'air_quality_distribution': df['air_quality'].value_counts().to_dict(),
            'feature_statistics': df[self.feature_columns].describe().to_dict(),
            'correlation_matrix': df[self.feature_columns].corr().to_dict()
        }
        return stats
    
    def generate_time_series_data(self, feature, time_period='1D'):
        """Generate time series data for a specific feature"""
        return self.df.set_index('timestamp')[feature].resample(time_period).mean()
    
    def get_pollution_hotspots(self, threshold=75):
        """Identify pollution hotspots based on PM2.5 levels"""
        return self.df[self.df['pm25'] > threshold][['latitude', 'longitude', 'pm25']]
    
    def calculate_health_risk(self, df=None):
        """Calculate health risk based on pollutant levels"""
        if df is None:
            df = self.df
            
        risk_factors = {
            'pm25': 0.3,
            'pm10': 0.2,
            'no2': 0.2,
            'so2': 0.15,
            'co': 0.15
        }
        
        risk_score = sum(
            df[pollutant] * weight 
            for pollutant, weight in risk_factors.items()
        )
        
        return risk_score 