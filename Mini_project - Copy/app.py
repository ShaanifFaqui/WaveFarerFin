"""
WaveFarer Beach Safety Prediction API
====================================

This Flask application provides REST API endpoints for beach safety predictions
and weather forecasting. It combines machine learning models with real-time
weather and marine data to provide safety recommendations.

Author: WaveFarer Team
Version: 1.0.0
"""

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import joblib
import json
from pymongo import MongoClient
from datetime import datetime
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import hashlib
import time
from dotenv import load_dotenv
from models.weather_forecast_model import WeatherForecastModel
from predict_weather import predict_weather_forecast

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variable configuration with defaults
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
PORT = int(os.environ.get("PORT", 5001))
HOST = os.environ.get("HOST", "0.0.0.0")
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

# Initialize Flask app with security settings
app = Flask(__name__)
app.config['SECRET_KEY'] = FLASK_SECRET_KEY
app.config['DEBUG'] = FLASK_DEBUG

# Configure CORS with security settings
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Database configuration with connection error handling
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test the connection
    client.admin.command('ping')
    print("✅ MongoDB connection successful")
    db = client['test']
    collection = db['predictions']
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    print("⚠️  Running without database functionality")
    client = None
    db = None
    collection = None

# Model configuration
INPUT_HOURS = 24
input_features = ['temperature', 'humidity', 'wind_direction', 'wind_speed']
target_features = ['temperature', 'wind_speed', 'wind_direction']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# MODEL LOADING
# =============================================================================

def load_models():
    """
    Load all required ML models and mappings.
    
    Returns:
        tuple: (alert_model, alert_mapping, safety_mapping, model_columns)
    """
    try:
        # Load alert prediction model
        alert_model = joblib.load('alert_model.pkl')
        
        # Load alert mappings
        with open('alert_mapping.json', 'r') as f:
            alert_mapping = json.load(f)
        code_to_alert = {int(v): k for k, v in alert_mapping.items()}
        
        # Load safety recommendations
        with open('alert_to_safety_mapping.json', 'r') as f:
            alert_to_safety = json.load(f)
        
        # Load model column specifications
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
            
        return alert_model, code_to_alert, alert_to_safety, model_columns
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Load models on startup
alert_model, code_to_alert, alert_to_safety, model_columns = load_models()

# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_and_prepare_data():
    """
    Load and preprocess weather data from CSV files.
    
    Returns:
        tuple: (processed_dataframe, scalers_dict)
    """
    DATA_DIR = "./weather_data"
    dfs = []
    
    try:
        for fname in os.listdir(DATA_DIR):
            if fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(DATA_DIR, fname), skiprows=3)
                df.columns = [
                    "time", "temperature", "humidity", "wind_direction", "wind_speed",
                    "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
                    "wave_height", "wave_direction", "wind_wave_height", "wind_wave_direction",
                    "current_velocity", "current_direction", "sea_surface_temp"
                ]
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                
                # Convert numeric columns
                for col in df.columns[1:]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(inplace=True)
                dfs.append(df)

        df = pd.concat(dfs).sort_values(by="time").reset_index(drop=True)

        # Normalize data
        scalers = {col: MinMaxScaler() for col in df.columns if col != 'time'}
        for col in scalers:
            df[col] = scalers[col].fit_transform(df[[col]])

        return df, scalers
        
    except Exception as e:
        print(f"Error loading weather data: {e}")
        raise

def prepare_input_data(data):
    """
    Prepare input data for prediction model.
    
    Args:
        data (dict): Raw input data from API request
        
    Returns:
        pd.DataFrame: Processed input data ready for prediction
    """
    input_data = {
        'Latitude': data.get('latitude', 0),
        'Longitude': data.get('longitude', 0),
        'Temperature (°C)': data.get('Temperature', 0),
        'Humidity (%)': data.get('Humidity', 0),
        'Wind Speed (m/s)': data.get('WindSpeed', 0),
        'Cloud Cover (%)': data.get('CloudCover', 0),
        'Wave Height (m)': data.get('WaveHeight', 0),
        'Ocean Current Velocity (m/s)': data.get('OceanCurrentVelocity', 0),
        'Sea Surface Temp (°C)': data.get('SeaSurfaceTemp', 0),
        'Weather Condition': data.get('WeatherCode', 0),
        'Beach Name': data.get('BeachName', '')
    }

    input_df = pd.DataFrame([input_data])
    categorical_cols = ['Beach Name', 'Weather Condition']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Ensure all required columns are present
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    return input_df[model_columns]

# =============================================================================
# WEATHER MODEL SETUP
# =============================================================================

def setup_weather_model():
    """
    Initialize and load the weather forecasting model.
    
    Returns:
        WeatherForecastModel: Loaded and configured weather model
    """
    try:
        df, scalers = load_and_prepare_data()
        weather_model = WeatherForecastModel(
            input_size=4, 
            hidden_size=128, 
            output_size=72 * 3
        ).to(device)
        
        weather_model.load_state_dict(
            torch.load("models/weather_forecast_model.pth", map_location=device)
        )
        weather_model.eval()
        
        return weather_model
        
    except Exception as e:
        print(f"Error setting up weather model: {e}")
        raise

# Initialize weather model
weather_model = setup_weather_model()

# =============================================================================
# SECURITY & RATE LIMITING
# =============================================================================

# Simple in-memory rate limiting (use Redis in production)
request_counts = {}

def check_rate_limit(ip_address: str, limit: int = 100, window: int = 3600) -> bool:
    """
    Simple rate limiting implementation.
    
    Args:
        ip_address (str): Client IP address
        limit (int): Maximum requests per window
        window (int): Time window in seconds
        
    Returns:
        bool: True if request is allowed
    """
    current_time = time.time()
    
    if ip_address not in request_counts:
        request_counts[ip_address] = {'count': 1, 'reset_time': current_time + window}
        return True
    
    # Reset counter if window has passed
    if current_time > request_counts[ip_address]['reset_time']:
        request_counts[ip_address] = {'count': 1, 'reset_time': current_time + window}
        return True
    
    # Check if limit exceeded
    if request_counts[ip_address]['count'] >= limit:
        return False
    
    request_counts[ip_address]['count'] += 1
    return True

def add_security_headers(response):
    """Add security headers to response."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict beach safety based on current conditions.
    
    Expected JSON payload:
    {
        "latitude": float,
        "longitude": float,
        "Temperature": float,
        "Humidity": float,
        "WindSpeed": float,
        "CloudCover": float,
        "WeatherCode": int,
        "WaveHeight": float,
        "OceanCurrentVelocity": float,
        "SeaSurfaceTemp": float,
        "BeachName": string,
        "user_mail": string
    }
    
    Returns:
        JSON response with alert and safety messages
    """
    try:
        # Rate limiting check
        client_ip = request.remote_addr
        if not check_rate_limit(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.json
        if not data:
            return jsonify({"error": "Empty request body"}), 400
        
        # Validate required fields
        required_fields = ['latitude', 'longitude', 'Temperature', 'Humidity', 'WindSpeed']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400
        
        # Validate coordinate ranges
        lat, lon = data.get('latitude'), data.get('longitude')
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error": "Invalid coordinates"}), 400
        
        print(f"Received prediction request for coordinates: ({lat}, {lon}) from {client_ip}")

        # Prepare input data
        input_df = prepare_input_data(data)
        
        # Make prediction
        predicted_code = alert_model.predict(input_df)[0]
        alert_message = code_to_alert[predicted_code]
        safety_message = alert_to_safety.get(alert_message, "No safety message available.")
        
        response = {
            'alert_message': alert_message,
            'safety_message': safety_message
        }
        
        # Store prediction in database (if available)
        if collection is not None:
            try:
                db_entry = {
                    "timestamp": datetime.utcnow(),
                    "email": data.get("user_mail", ""),
                    "BeachName": data.get("BeachName", ""),
                    "input": data,
                    "prediction": {
                        "alert_message": alert_message,
                        "safety_message": safety_message
                    }
                }
                collection.insert_one(db_entry)
                print("✅ Prediction stored in database")
            except Exception as db_error:
                print(f"⚠️  Database storage failed: {db_error}")
        else:
            print("⚠️  Database not available - skipping storage")

        response_obj = jsonify(response)
        return add_security_headers(response_obj)

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        return jsonify({"message": error_msg, "status": "error"}), 500


@app.route('/api/future-predict', methods=['POST'])
def forecast():
    """
    Get 3-day weather forecast for a specific location.
    
    Expected JSON payload:
    {
        "lat": float,
        "lon": float
    }
    
    Returns:
        JSON response with 3-day forecast data
    """
    try:
        # Rate limiting check
        client_ip = request.remote_addr
        if not check_rate_limit(client_ip):
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Empty request body"}), 400
        
        # Validate input
        if not data or 'lat' not in data or 'lon' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing latitude or longitude in request body."
            }), 400

        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        
        # Validate coordinate ranges
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({"error": "Invalid coordinates"}), 400
        
        # Get forecast
        forecast_data = predict_weather_forecast(lat, lon)
        
        return jsonify({
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "forecast": forecast_data
        })
        
    except Exception as e:
        error_msg = f"Forecast error: {str(e)}"
        print(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API status.
    
    Returns:
        JSON response with API status
    """
    return jsonify({
        "status": "healthy",
        "message": "WaveFarer API is running",
        "timestamp": datetime.utcnow().isoformat()
    })


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("🌊 Starting WaveFarer Beach Safety API...")
    print(f"📊 Using device: {device}")
    print(f"🔧 Configuration:")
    print(f"   - Host: {HOST}")
    print(f"   - Port: {PORT}")
    print(f"   - Debug: {FLASK_DEBUG}")
    print(f"   - Database: {'Connected' if client else 'Not available'}")
    print(f"🚀 Server starting on http://{HOST}:{PORT}")
    app.run(debug=FLASK_DEBUG, host=HOST, port=PORT)
