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
        self.data_processor = AirQualityDataProcessor()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize Gemini API
        load_dotenv()
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        else:
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
        """Train the model with preprocessed data"""
        try:
            X, y = self.data_processor.prepare_training_data()
            if X is not None and y is not None and len(X) > 0 and len(y) > 0:
                # Convert categorical labels to numeric values
                quality_map = {
                    'Good': 0,
                    'Moderate': 1,
                    'Poor': 2,
                    'Hazardous': 3
                }
                y_numeric = np.array([quality_map[cat] for cat in y])
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y_numeric)
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def predict_air_quality(self, features):
        """Make predictions for new data"""
        if not self.is_trained:
            if not self.train_model():
                return None, None
        
        try:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Convert numeric prediction back to category
            quality_map = {
                0: 'Good',
                1: 'Moderate',
                2: 'Poor',
                3: 'Hazardous'
            }
            prediction = quality_map[prediction[0]]
            
            return prediction, probabilities[0]
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None, None
    
    def get_ai_insights(self, features, prediction, probabilities):
        """Generate AI insights using Gemini"""
        if not self.gemini_model:
            return "AI insights are currently unavailable. Please check the API configuration."
        
        try:
            prompt = f"""
            Based on the following air quality data and prediction:
            Features: {features}
            Predicted Air Quality: {prediction}
            Probabilities: {probabilities}
            
            Please provide:
            1. A brief explanation of the prediction
            2. Health implications
            3. Recommended actions
            """
            
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