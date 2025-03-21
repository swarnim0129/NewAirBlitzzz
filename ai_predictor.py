import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from data_processor import AirQualityDataProcessor
import os
from dotenv import load_dotenv

class AirQualityPredictor:
    def __init__(self):
        load_dotenv()
        self.data_processor = AirQualityDataProcessor()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Configure Gemini AI
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        if GEMINI_API_KEY:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                print(f"Error configuring Gemini API: {str(e)}")
                self.gemini_model = None
        else:
            print("GEMINI_API_KEY not found in environment variables")
            self.gemini_model = None
        
        # Define air quality categories and their numeric values
        self.quality_categories = {
            'Good': 0,
            'Moderate': 1,
            'Poor': 2,
            'Hazardous': 3
        }
        self.reverse_categories = {v: k for k, v in self.quality_categories.items()}
        
    def train_model(self):
        """Train the Random Forest model"""
        X, y = self.data_processor.prepare_training_data()
        # Convert categorical labels to numeric values
        y_numeric = np.array([self.quality_categories[cat] for cat in y])
        self.model.fit(X, y_numeric)
        self.is_trained = True
        return self.model
    
    def predict_air_quality(self, features):
        """Predict air quality for given features"""
        if not self.is_trained:
            self.train_model()
            
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Make prediction
        prediction = self.model.predict(scaled_features)
        probabilities = self.model.predict_proba(scaled_features)
        
        # Convert numeric prediction to category
        predicted_category = self.reverse_categories[prediction[0]]
        
        return predicted_category, probabilities[0]
    
    def get_ai_insights(self, data):
        """Get AI-powered insights using Gemini"""
        if not self.gemini_model:
            return "AI insights are currently unavailable. Please check the API configuration."
            
        prompt = f"""
        Analyze this air quality data and provide insights:
        Temperature: {data['temperature']}°C
        Humidity: {data['humidity']}%
        PM2.5: {data['pm25']} µg/m³
        PM10: {data['pm10']} µg/m³
        NO2: {data['no2']} ppb
        SO2: {data['so2']} ppb
        CO: {data['co']} ppm
        Industrial Proximity: {data['industrial_proximity']} km
        Population Density: {data['population_density']} people/km²
        
        Please provide:
        1. Main pollution sources
        2. Health risks
        3. Recommendations for improvement
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def predict_future_quality(self, current_data, hours_ahead=24):
        """Predict air quality for future hours"""
        if not self.is_trained:
            self.train_model()
            
        # Create future timestamps
        future_times = pd.date_range(
            start=current_data['timestamp'],
            periods=hours_ahead,
            freq='H'
        )
        
        # Generate predictions for each hour
        predictions = []
        for time in future_times:
            # Adjust features based on time of day
            hour = time.hour
            features = current_data.copy()
            
            # Add time-based adjustments
            features['temperature'] += np.sin(hour * np.pi / 12) * 2  # Daily temperature cycle
            features['humidity'] += np.cos(hour * np.pi / 12) * 5    # Daily humidity cycle
            
            # Make prediction
            prediction, probability = self.predict_air_quality(
                pd.DataFrame([features]).drop(['timestamp', 'air_quality'], axis=1)
            )
            
            predictions.append({
                'timestamp': time,
                'predicted_quality': prediction,
                'confidence': max(probability)
            })
        
        return pd.DataFrame(predictions)
    
    def get_recommendations(self, current_data):
        """Get AI-powered recommendations for improving air quality"""
        if not self.gemini_model:
            return "AI recommendations are currently unavailable. Please check the API configuration."
            
        prompt = f"""
        Based on these air quality parameters:
        Temperature: {current_data['temperature']}°C
        Humidity: {current_data['humidity']}%
        PM2.5: {current_data['pm25']} µg/m³
        PM10: {current_data['pm10']} µg/m³
        NO2: {current_data['no2']} ppb
        SO2: {current_data['so2']} ppb
        CO: {current_data['co']} ppm
        
        Please provide specific, actionable recommendations for:
        1. Immediate actions to improve air quality
        2. Long-term solutions
        3. Health precautions for sensitive groups
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def analyze_trends(self, historical_data):
        """Analyze trends and patterns in historical data"""
        if not self.gemini_model:
            return "Trend analysis is currently unavailable. Please check the API configuration."
            
        prompt = f"""
        Analyze these air quality trends:
        {historical_data.describe().to_string()}
        
        Please provide:
        1. Key patterns and trends
        2. Potential causes
        3. Seasonal variations
        4. Recommendations for monitoring
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing trends: {str(e)}" 