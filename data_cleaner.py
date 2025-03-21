import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def clean_data():
    # Read the original dataset
    df = pd.read_csv('TechBlitz DataScience Dataset.csv')
    
    # Rename columns to match our application's format
    column_mapping = {
        'PM2.5': 'pm25',
        'PM10': 'pm10',
        'NO2': 'no2',
        'SO2': 'so2',
        'CO': 'co',
        'Proximity_to_Industrial_Areas': 'industrial_proximity',
        'Population_Density': 'population_density',
        'Air Quality': 'air_quality'
    }
    df = df.rename(columns=column_mapping)
    
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Add timestamp column (last 1000 hours from now)
    now = datetime.now()
    df['timestamp'] = [now - timedelta(hours=i) for i in range(len(df))]
    
    # Add random latitude and longitude for demonstration
    # Centering around New York City coordinates
    df['latitude'] = np.random.normal(40.7128, 0.1, len(df))
    df['longitude'] = np.random.normal(-74.0060, 0.1, len(df))
    
    # Clean and validate data
    # Remove any rows with negative values
    for col in ['temperature', 'humidity', 'pm25', 'pm10', 'no2', 'so2', 'co', 
                'industrial_proximity', 'population_density']:
        df = df[df[col] >= 0]
    
    # Ensure humidity is between 0 and 100
    df['humidity'] = df['humidity'].clip(0, 100)
    
    # Remove any extreme outliers (values beyond 3 standard deviations)
    numeric_columns = ['temperature', 'pm25', 'pm10', 'no2', 'so2', 'co', 
                      'industrial_proximity', 'population_density']
    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[abs(df[col] - mean) <= 3 * std]
    
    # Ensure air quality categories are standardized
    df['air_quality'] = df['air_quality'].str.title()
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Save the cleaned dataset
    df.to_csv('finaldata.csv', index=False)
    print(f"Cleaned data saved to finaldata.csv ({len(df)} rows)")
    
    # Display some basic statistics
    print("\nBasic statistics:")
    print(f"Number of samples: {len(df)}")
    print("\nAir quality distribution:")
    print(df['air_quality'].value_counts())
    
if __name__ == "__main__":
    clean_data() 