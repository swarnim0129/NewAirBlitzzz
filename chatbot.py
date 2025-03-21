import google.generativeai as genai
from data_processor import AirQualityDataProcessor
import os
from dotenv import load_dotenv
import streamlit as st

class AirQualityChatbot:
    def __init__(self):
        load_dotenv()
        self.data_processor = AirQualityDataProcessor()
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.model = None
        self.initialize_model()
        
        # Initialize chat history
        self.chat_history = []
        
    def initialize_model(self):
        """Initialize the Gemini model with proper error handling"""
        if not self.GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not found in environment variables")
            return
            
        try:
            genai.configure(api_key=self.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            st.error(f"Error initializing Gemini model: {str(e)}")
        
    def get_response(self, user_input):
        """Get response from the chatbot"""
        if not self.model:
            return "I apologize, but I'm currently unavailable. Please check the API configuration."
            
        try:
            response = self.model.generate_content(user_input)
            return response.text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def _get_context(self):
        """Get current context about air quality data"""
        try:
            latest_data = self.data_processor.get_latest_data(n_samples=1).iloc[0]
            stats = self.data_processor.get_statistics()
            
            context = f"""
            Current Air Quality Status:
            - Temperature: {latest_data['temperature']}°C
            - Humidity: {latest_data['humidity']}%
            - PM2.5: {latest_data['pm25']} µg/m³
            - PM10: {latest_data['pm10']} µg/m³
            - NO2: {latest_data['no2']} ppb
            - SO2: {latest_data['so2']} ppb
            - CO: {latest_data['co']} ppm
            - Air Quality Category: {latest_data['air_quality']}
            
            Overall Statistics:
            - Total Samples: {stats['total_samples']}
            - Air Quality Distribution: {stats['air_quality_distribution']}
            """
            
            return context
        except Exception as e:
            st.error(f"Error getting context: {str(e)}")
            return "Error retrieving air quality data."
    
    def get_chat_history(self):
        """Get the chat history"""
        return self.chat_history
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history = []
    
    def get_quick_facts(self):
        """Get quick facts about air quality"""
        if not self.model:
            return "Quick facts are currently unavailable. Please check the API configuration."
            
        prompt = "Please provide 3 interesting facts about air quality and its impact on health."
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating quick facts: {str(e)}"
    
    def get_health_advice(self):
        """Get health advice based on current air quality"""
        if not self.model:
            return "Sorry, the AI model is not properly configured. Please check the API key and try again."
            
        try:
            latest_data = self.data_processor.get_latest_data(n_samples=1).iloc[0]
            
            prompt = f"""
            Based on these air quality parameters:
            Temperature: {latest_data['temperature']}°C
            Humidity: {latest_data['humidity']}%
            PM2.5: {latest_data['pm25']} µg/m³
            PM10: {latest_data['pm10']} µg/m³
            NO2: {latest_data['no2']} ppb
            SO2: {latest_data['so2']} ppb
            CO: {latest_data['co']} ppm
            Air Quality: {latest_data['air_quality']}
            
            Please provide:
            1. Health risks for the general population
            2. Special precautions for sensitive groups
            3. Recommended activities or restrictions
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating health advice: {str(e)}")
            return "Sorry, there was an error generating health advice. Please try again."
    
    def get_recommendations(self, pollutant_levels):
        """Get personalized recommendations based on pollutant levels"""
        if not self.model:
            return "Recommendations are currently unavailable. Please check the API configuration."
            
        prompt = f"""
        Based on these pollutant levels:
        PM2.5: {pollutant_levels.get('pm25', 'N/A')} µg/m³
        PM10: {pollutant_levels.get('pm10', 'N/A')} µg/m³
        NO2: {pollutant_levels.get('no2', 'N/A')} ppb
        SO2: {pollutant_levels.get('so2', 'N/A')} ppb
        CO: {pollutant_levels.get('co', 'N/A')} ppm
        
        Please provide:
        1. Health risks associated with these levels
        2. Immediate actions to take
        3. Long-term recommendations
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating recommendations: {str(e)}" 