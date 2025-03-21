import streamlit as st

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_option_menu import option_menu
from streamlit_extras.colored_header import colored_header
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Import our custom modules
from data_processor import AirQualityDataProcessor
from visualizations import AirQualityVisualizer
from ai_predictor import AirQualityPredictor
from chatbot import AirQualityChatbot

# Load environment variables
load_dotenv()

# Initialize components
data_processor = AirQualityDataProcessor()
visualizer = AirQualityVisualizer()
predictor = AirQualityPredictor()
chatbot = AirQualityChatbot()

# Custom CSS for professional theme
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #1E1E1E;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stSlider>div>div>div {
        background-color: #1E88E5;
    }
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        color: #1E1E1E;
        border: 1px solid #E0E0E0;
    }
    .stSelectbox>div>div>select {
        background-color: #FFFFFF;
        color: #1E1E1E;
        border: 1px solid #E0E0E0;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stMarkdown {
        color: #1E1E1E;
    }
    .stMetric {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    /* General styles */
    .stApp {
        background-color: white;
        color: #333;
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Card styles */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chatbot styles */
    .chatbot-container {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chatbot-header {
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #eee;
    }
    
    .chatbot-messages {
        max-height: 400px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        margin-right: 2rem;
    }
    
    .message-content {
        color: #333;
    }
    
    .quick-facts {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .fact-item {
        background-color: white;
        padding: 0.75rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Button styles */
    .stButton>button {
        background-color: #2196F3;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #1976D2;
    }
    
    /* Input styles */
    .stTextInput>div>div>input {
        border-radius: 4px;
        border: 1px solid #ddd;
    }
    
    /* Metric styles */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2196F3;
    }
    
    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🌍 Air Quality Dashboard")
    selected = option_menu(
        menu_title="Navigation",
        options=["Dashboard", "AI Predictions", "3D Visualizations", "Chatbot", "Reports"],
        icons=["house", "graph-up", "cube", "chat", "file-text"],
        menu_icon="cast",
        default_index=0,
    )

# Main content area
if selected == "Dashboard":
    colored_header(
        label="Air Quality Overview",
        description="Real-time monitoring and analysis of air quality parameters",
        color_name="blue-70"
    )
    
    # Get latest data
    latest_data = data_processor.get_latest_data(n_samples=1).iloc[0]
    stats = data_processor.get_statistics()
    
    # Display key metrics in a grid
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Air Quality", latest_data['air_quality'])
    with col2:
        st.metric("Temperature", f"{latest_data['temperature']:.1f}°C")
    with col3:
        st.metric("PM2.5", f"{latest_data['pm25']:.1f} µg/m³")
    with col4:
        st.metric("Humidity", f"{latest_data['humidity']:.1f}%")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pollutant Trends")
        pollutant = st.selectbox(
            "Select Pollutant",
            ['pm25', 'pm10', 'no2', 'so2', 'co']
        )
        fig = visualizer.create_animated_line_chart(pollutant)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Pollutant Distribution")
        fig = visualizer.create_3d_boxplot(pollutant)
        st.plotly_chart(fig, use_container_width=True)
    
    # Full-width heatmap
    st.subheader("Pollutant Correlation Heatmap")
    fig = visualizer.create_heatmap()
    st.plotly_chart(fig, use_container_width=True)
    
    # AI insights in a card-like container
    st.subheader("AI-Powered Insights")
    with st.container():
        st.markdown("""
            <div style='background-color: #F5F5F5; padding: 1rem; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
        """, unsafe_allow_html=True)
        try:
            # Get prediction and probabilities first
            features = pd.DataFrame([latest_data])
            prediction, probabilities = predictor.predict_air_quality(features)
            if prediction is not None and probabilities is not None:
                insights = predictor.get_ai_insights(features.iloc[0], prediction, probabilities)
                st.write(insights)
            else:
                st.error("Unable to generate insights. Please try again later.")
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)

elif selected == "AI Predictions":
    colored_header(
        label="AI-Powered Predictions",
        description="Predict air quality using advanced AI models",
        color_name="blue-70"
    )
    
    # Input parameters in a grid
    st.subheader("Enter Parameters for Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 50.0)
        pm10 = st.slider("PM10 (µg/m³)", 0.0, 500.0, 75.0)
    
    with col2:
        no2 = st.slider("NO2 (ppb)", 0.0, 500.0, 40.0)
        so2 = st.slider("SO2 (ppb)", 0.0, 500.0, 20.0)
        co = st.slider("CO (ppm)", 0.0, 10.0, 2.0)
        industrial_proximity = st.slider("Industrial Proximity (km)", 0.0, 20.0, 5.0)
    
    # Make prediction
    if st.button("Predict Air Quality"):
        try:
            features = pd.DataFrame([{
                'temperature': temperature,
                'humidity': humidity,
                'pm25': pm25,
                'pm10': pm10,
                'no2': no2,
                'so2': so2,
                'co': co,
                'industrial_proximity': industrial_proximity,
                'population_density': 1000,  # Default value
                'latitude': 40.7128,  # Default value
                'longitude': -74.0060  # Default value
            }])
            
            prediction, probabilities = predictor.predict_air_quality(features)
            
            if prediction is not None and probabilities is not None:
                # Display results in a card-like container
                with st.container():
                    st.markdown("""
                        <div style='background-color: #F5F5F5; padding: 1rem; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    """, unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Air Quality", prediction)
                    
                    with col2:
                        confidence = max(probabilities) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display recommendations
                st.subheader("Recommendations")
                with st.container():
                    st.markdown("""
                        <div style='background-color: #F5F5F5; padding: 1rem; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    """, unsafe_allow_html=True)
                    recommendations = predictor.get_recommendations(features.iloc[0])
                    st.write(recommendations)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Unable to make prediction. Please check the input parameters and try again.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # 3D Boxplot
    st.subheader("3D Boxplot of Pollutant Levels")
    pollutant = st.selectbox(
        "Select Pollutant for Boxplot",
        ['pm25', 'pm10', 'no2', 'so2', 'co']
    )
    fig = visualizer.create_3d_boxplot(pollutant)
    st.plotly_chart(fig, use_container_width=True)
    
    # Force-directed graph
    st.subheader("Pollutant Relationship Network")
    fig = visualizer.create_force_directed_graph()
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D Bar chart
    st.subheader("3D Bar Chart of Average Pollutant Levels")
    fig = visualizer.create_3d_bar_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    # Add new 3D visualizations
    st.subheader("3D Surface Plot of Pollution Levels")
    pollutant = st.selectbox(
        "Select Pollutant for Surface Plot",
        ['pm25', 'pm10', 'no2', 'so2', 'co']
    )
    param1 = st.selectbox(
        "Select First Parameter for Surface",
        ['temperature', 'humidity', 'industrial_proximity']
    )
    param2 = st.selectbox(
        "Select Second Parameter for Surface",
        ['temperature', 'humidity', 'industrial_proximity']
    )
    if param1 != param2:
        try:
            fig = visualizer.create_3d_surface_plot(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating surface plot: {str(e)}")
    
    st.subheader("3D Scatter Plot of Pollutants")
    pollutant = st.selectbox(
        "Select Pollutant for Scatter Plot",
        ['pm25', 'pm10', 'no2', 'so2', 'co']
    )
    param1 = st.selectbox(
        "Select First Parameter for Scatter",
        ['temperature', 'humidity', 'industrial_proximity']
    )
    param2 = st.selectbox(
        "Select Second Parameter for Scatter",
        ['temperature', 'humidity', 'industrial_proximity']
    )
    if param1 != param2:
        try:
            fig = visualizer.create_3d_scatter_plot(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter plot: {str(e)}")
    
    # Correlation heatmap
    st.subheader("Pollutant Correlation Heatmap")
    fig = visualizer.create_heatmap()
    st.plotly_chart(fig, use_container_width=True)

elif selected == "3D Visualizations":
    colored_header(
        label="3D Pollution Visualizations",
        description="Interactive 3D representations of air quality data",
        color_name="blue-70"
    )
    
    # 3D Boxplot
    st.subheader("3D Boxplot of Pollutant Levels")
    pollutant = st.selectbox(
        "Select Pollutant for Boxplot",
        ['pm25', 'pm10', 'no2', 'so2', 'co']
    )
    fig = visualizer.create_3d_boxplot(pollutant)
    st.plotly_chart(fig, use_container_width=True)
    
    # Force-directed graph
    st.subheader("Pollutant Relationship Network")
    fig = visualizer.create_force_directed_graph()
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D Bar chart
    st.subheader("3D Bar Chart of Average Pollutant Levels")
    fig = visualizer.create_3d_bar_chart()
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Pollutant Correlation Heatmap")
    fig = visualizer.create_heatmap()
    st.plotly_chart(fig, use_container_width=True)

elif selected == "Chatbot":
    colored_header(
        label="AI Air Quality Assistant",
        description="Ask questions about air quality and get AI-powered insights",
        color_name="blue-70"
    )
    
    # Chatbot section
    st.subheader("AI Assistant")
    st.markdown("""
        <div class="chatbot-container">
            <div class="chatbot-header">
                <h4>Ask me anything about air quality!</h4>
            </div>
            <div class="chatbot-messages">
    """, unsafe_allow_html=True)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="user-message">
                        <div class="message-content">{message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="assistant-message">
                        <div class="message-content">{message["content"]}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Type your message here...")
    if st.button("Send"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get AI response
            response = chatbot.generate_response(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update chat display
            st.rerun()
    
    # Quick facts
    st.markdown("""
        <div class="quick-facts">
            <h4>Quick Facts</h4>
            <div class="facts-content">
    """, unsafe_allow_html=True)
    
    try:
        facts = chatbot.get_quick_facts()
        st.markdown(f"<div class='fact-item'>{facts}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error generating quick facts: {str(e)}")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

elif selected == "Reports":
    colored_header(
        label="Air Quality Reports",
        description="Generate and download comprehensive air quality reports",
        color_name="blue-70"
    )
    
    # Generate report
    if st.button("Generate Report"):
        # Get data
        latest_data = data_processor.get_latest_data(n_samples=100)
        stats = data_processor.get_statistics()
        
        # Create report content
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_air_quality": latest_data.iloc[0]['air_quality'],
            "statistics": stats,
            "trends": predictor.analyze_trends(latest_data),
            "recommendations": predictor.get_recommendations(latest_data.iloc[0])
        }
        
        # Display report in a card-like container
        st.subheader("Air Quality Report")
        with st.container():
            st.markdown("""
                <div style='background-color: #F5F5F5; padding: 1rem; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
            """, unsafe_allow_html=True)
            st.write("Current Status:", report["current_air_quality"])
            st.write("Statistics:", report["statistics"])
            st.write("Trends Analysis:", report["trends"])
            st.write("Recommendations:", report["recommendations"])
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            label="Download Report",
            data=json.dumps(report, indent=2),
            file_name="air_quality_report.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>🌍 Air Quality Dashboard | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def show_visualizations_page():
    st.title("Air Quality Visualizations")
    
    # Add tabs for different visualization types
    tab1, tab2, tab3 = st.tabs(["Basic Plots", "Advanced Plots", "3D Visualizations"])
    
    with tab1:
        # Basic plots
        st.header("Basic Plots")
        
        # Time series plot
        st.subheader("Time Series Plot")
        pollutant = st.selectbox("Select Pollutant for Time Series", visualizer.pollutants)
        fig = visualizer.create_time_series_plot(pollutant)
        st.plotly_chart(fig, use_container_width=True)
        
        # Box plot
        st.subheader("Box Plot")
        fig = visualizer.create_box_plot()
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.subheader("Scatter Plot")
        pollutant = st.selectbox("Select Pollutant for Scatter", visualizer.pollutants)
        param = st.selectbox("Select Parameter", visualizer.parameters)
        fig = visualizer.create_scatter_plot(pollutant, param)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Advanced plots
        st.header("Advanced Plots")
        
        # Heatmap
        st.subheader("Correlation Heatmap")
        fig = visualizer.create_heatmap()
        st.plotly_chart(fig, use_container_width=True)
        
        # Force-directed graph
        st.subheader("Force-Directed Graph")
        fig = visualizer.create_force_directed_graph()
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic visualization
        st.subheader("Geographic Visualization")
        pollutant = st.selectbox("Select Pollutant for Map", visualizer.pollutants)
        fig = visualizer.create_geographic_visualization(pollutant)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("3D Visualizations")
        
        # 3D Surface Plot
        st.subheader("3D Surface Plot")
        pollutant = st.selectbox("Select Pollutant", visualizer.pollutants)
        param1 = st.selectbox("Select First Parameter", visualizer.parameters)
        param2 = st.selectbox("Select Second Parameter", visualizer.parameters)
        if param1 != param2:
            fig = visualizer.create_3d_surface_plot(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Scatter Plot
        st.subheader("3D Scatter Plot")
        pollutant = st.selectbox("Select Pollutant for Scatter", visualizer.pollutants)
        param1 = st.selectbox("Select First Parameter for Scatter", visualizer.parameters)
        param2 = st.selectbox("Select Second Parameter for Scatter", visualizer.parameters)
        if param1 != param2:
            fig = visualizer.create_3d_scatter_plot(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Bar Plot
        st.subheader("3D Bar Plot")
        pollutant = st.selectbox("Select Pollutant for Bar Plot", visualizer.pollutants)
        fig = visualizer.create_3d_bar_plot(pollutant)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Density Plot
        st.subheader("3D Density Plot")
        pollutant = st.selectbox("Select Pollutant for Density", visualizer.pollutants)
        param1 = st.selectbox("Select First Parameter for Density", visualizer.parameters)
        param2 = st.selectbox("Select Second Parameter for Density", visualizer.parameters)
        if param1 != param2:
            fig = visualizer.create_3d_density_plot(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Vector Field
        st.subheader("3D Vector Field")
        pollutant = st.selectbox("Select Pollutant for Vector Field", visualizer.pollutants)
        param1 = st.selectbox("Select First Parameter for Vector Field", visualizer.parameters)
        param2 = st.selectbox("Select Second Parameter for Vector Field", visualizer.parameters)
        if param1 != param2:
            fig = visualizer.create_3d_vector_field(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Network Graph
        st.subheader("3D Network Graph")
        fig = visualizer.create_3d_network_graph()
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Contour Plot
        st.subheader("3D Contour Plot")
        pollutant = st.selectbox("Select Pollutant for Contour", visualizer.pollutants)
        param1 = st.selectbox("Select First Parameter for Contour", visualizer.parameters)
        param2 = st.selectbox("Select Second Parameter for Contour", visualizer.parameters)
        if param1 != param2:
            fig = visualizer.create_3d_contour_plot(pollutant, param1, param2)
            st.plotly_chart(fig, use_container_width=True)
        
        # Ternary Plot
        st.subheader("Ternary Plot")
        pollutant1 = st.selectbox("Select First Pollutant", visualizer.pollutants)
        pollutant2 = st.selectbox("Select Second Pollutant", visualizer.pollutants)
        pollutant3 = st.selectbox("Select Third Pollutant", visualizer.pollutants)
        if pollutant1 != pollutant2 and pollutant2 != pollutant3 and pollutant1 != pollutant3:
            fig = visualizer.create_3d_ternary_plot(pollutant1, pollutant2, pollutant3)
            st.plotly_chart(fig, use_container_width=True) 