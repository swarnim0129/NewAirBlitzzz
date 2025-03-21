# Air Quality Dashboard

A comprehensive air quality monitoring and analysis dashboard built with Streamlit, featuring real-time monitoring, AI predictions, and interactive visualizations.

## Features

- Real-time air quality monitoring
- AI-powered predictions and insights
- Interactive 3D visualizations
- Chatbot assistant for air quality information
- Comprehensive reports generation
- Professional white theme UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/swarnim0129/TechBlitz.git
cd TechBlitz
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your API keys:
```
GEMINI_API_KEY=your_gemini_api_key
```

5. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `data_processor.py`: Data processing and analysis
- `visualizations.py`: Visualization components
- `ai_predictor.py`: AI prediction models
- `chatbot.py`: AI chatbot implementation

## Dependencies

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- google-generativeai
- python-dotenv
- folium
- seaborn
- matplotlib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

