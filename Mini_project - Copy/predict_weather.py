"""
Weather Forecasting Module for WaveFarer
=======================================

This module provides weather forecasting capabilities using LSTM neural networks.
It processes historical weather data and predicts future weather conditions
for beach safety assessment.

Author: WaveFarer Team
Version: 1.0.0
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configuration
INPUT_FEATURES = ['temperature', 'humidity', 'wind_direction', 'wind_speed']
TARGET_FEATURES = ['temperature', 'wind_speed', 'wind_direction', 'wave_height', 'sea_surface_temp']
INPUT_HOURS = 24
OUTPUT_HOURS = 72
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# NEURAL NETWORK MODEL
# =============================================================================

class WeatherForecastModel(nn.Module):
    """
    LSTM-based weather forecasting model.
    
    This model uses LSTM layers to predict weather conditions based on
    historical weather data. It takes 24 hours of input and predicts
    72 hours of weather conditions.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        """
        Initialize the weather forecasting model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM
            output_size (int): Number of output features
            num_layers (int): Number of LSTM layers
        """
        super(WeatherForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, OUTPUT_HOURS, num_features)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.view(-1, OUTPUT_HOURS, len(TARGET_FEATURES))

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def initialize_model() -> WeatherForecastModel:
    """
    Initialize and load the weather forecasting model.
    
    Returns:
        WeatherForecastModel: Loaded and configured model
    """
    try:
        model = WeatherForecastModel(
            input_size=len(INPUT_FEATURES),
            hidden_size=128,
            output_size=OUTPUT_HOURS * len(TARGET_FEATURES)
        ).to(DEVICE)
        
        model.load_state_dict(torch.load("models/weather_forecast_model_v2.pth", map_location=DEVICE))
        model.eval()
        
        print(f"✅ Weather model loaded successfully on {DEVICE}")
        return model
        
    except Exception as e:
        print(f"❌ Error loading weather model: {e}")
        raise

# Initialize model globally
MODEL = initialize_model()

# =============================================================================
# DATA PROCESSING
# =============================================================================

def load_weather_data() -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """
    Load and preprocess weather data from CSV files.
    
    Returns:
        Tuple[pd.DataFrame, Dict]: Processed dataframe and scalers dictionary
    """
    DATA_DIR = "./weather_data"
    dfs = []
    
    try:
        print("📊 Loading weather data...")
        
        for fname in os.listdir(DATA_DIR):
            if fname.endswith(".csv"):
                file_path = os.path.join(DATA_DIR, fname)
                df = pd.read_csv(file_path, skiprows=3)
                
                # Define column names
                df.columns = [
                    "time", "temperature", "humidity", "wind_direction", "wind_speed",
                    "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
                    "wave_height", "wave_direction", "wind_wave_height", "wind_wave_direction",
                    "current_velocity", "current_direction", "sea_surface_temp"
                ]
                
                # Convert time column
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
                
                # Convert numeric columns
                for col in df.columns[1:]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(inplace=True)
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No CSV files found in weather_data directory")
        
        # Combine all dataframes
        df = pd.concat(dfs).sort_values(by="time").reset_index(drop=True)
        
        # Normalize data
        scalers = {col: MinMaxScaler() for col in df.columns if col != 'time'}
        for col in scalers:
            df[col] = scalers[col].fit_transform(df[[col]])
        
        print(f"✅ Loaded {len(df)} weather records")
        return df, scalers
        
    except Exception as e:
        print(f"❌ Error loading weather data: {e}")
        raise

def prepare_input_sequence(df: pd.DataFrame) -> torch.Tensor:
    """
    Prepare input sequence for the model.
    
    Args:
        df (pd.DataFrame): Processed weather dataframe
        
    Returns:
        torch.Tensor: Input tensor ready for model prediction
    """
    input_seq = df[INPUT_FEATURES].iloc[-INPUT_HOURS:].values
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return input_tensor

def add_location_variation(input_tensor: torch.Tensor, latitude: float, longitude: float) -> torch.Tensor:
    """
    Add location-based variation to input data.
    
    Args:
        input_tensor (torch.Tensor): Input tensor
        latitude (float): Location latitude
        longitude (float): Location longitude
        
    Returns:
        torch.Tensor: Modified input tensor with location variation
    """
    # Generate location-specific seed
    seed = int((latitude + 90) * 1000 + (longitude + 180))
    np.random.seed(seed)
    
    # Add small noise based on location
    noise = np.random.normal(0, 0.02, input_tensor.shape)
    input_tensor += torch.tensor(noise, dtype=torch.float32).to(DEVICE)
    
    return input_tensor

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def make_prediction(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Make weather prediction using the loaded model.
    
    Args:
        input_tensor (torch.Tensor): Prepared input tensor
        
    Returns:
        np.ndarray: Raw prediction output
    """
    with torch.no_grad():
        prediction = MODEL(input_tensor)
    
    return prediction.squeeze(0).cpu().numpy()

def denormalize_predictions(prediction_np: np.ndarray, scalers: Dict[str, MinMaxScaler]) -> np.ndarray:
    """
    Denormalize prediction values back to original scale.
    
    Args:
        prediction_np (np.ndarray): Normalized predictions
        scalers (Dict): Dictionary of scalers for each feature
        
    Returns:
        np.ndarray: Denormalized predictions
    """
    denorm_pred = np.zeros_like(prediction_np)
    
    for i, feature in enumerate(TARGET_FEATURES):
        if feature in scalers:
            denorm_pred[:, i] = scalers[feature].inverse_transform(
                prediction_np[:, i].reshape(-1, 1)
            ).flatten()
    
    return denorm_pred

def generate_daily_forecasts(denorm_pred: np.ndarray) -> List[Dict]:
    """
    Generate daily averaged forecasts from hourly predictions.
    
    Args:
        denorm_pred (np.ndarray): Denormalized predictions
        
    Returns:
        List[Dict]: List of daily forecast dictionaries
    """
    forecast_output = []
    
    for day in range(3):
        # Get 24 hours of data for this day
        day_data = denorm_pred[day * 24:(day + 1) * 24]
        
        # Calculate daily averages
        avg_temp = float(np.mean(day_data[:, 0]))
        avg_wind_spd = float(np.mean(day_data[:, 1]))
        avg_wind_dir = float(np.mean(day_data[:, 2]))
        avg_wave = float(np.mean(day_data[:, 3]))
        avg_sea_temp = float(np.mean(day_data[:, 4]))
        
        # Determine beach safety recommendation
        if avg_wave < 2 and avg_wind_spd < 15:
            beach_msg = "✅ Safe to go to the beach!"
        else:
            beach_msg = "⚠️ Not suitable for beach activities."
        
        # Create forecast entry
        forecast_entry = {
            "day": day + 1,
            "avg_temp": round(avg_temp, 2),
            "avg_wind_speed": round(avg_wind_spd, 2),
            "avg_wind_direction": round(avg_wind_dir, 2),
            "avg_wave_height": round(avg_wave, 2),
            "avg_sea_surface_temp": round(avg_sea_temp, 2),
            "beach_safety": beach_msg
        }
        
        forecast_output.append(forecast_entry)
        
        # Print forecast summary
        print(f"\n📅 Day {day+1}:")
        print(f"   🌡️  Avg Temp: {avg_temp:.2f}°C")
        print(f"   💨 Avg Wind: {avg_wind_spd:.2f} m/s, Direction: {avg_wind_dir:.2f}°")
        print(f"   🌊 Avg Wave Height: {avg_wave:.2f} m")
        print(f"   🌊 Sea Surface Temp: {avg_sea_temp:.2f}°C")
        print(f"   🏖️  Recommendation: {beach_msg}")
    
    return forecast_output

# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_weather_forecast(latitude: float, longitude: float) -> List[Dict]:
    """
    Predict 3-day weather forecast for a specific location.
    
    This function loads weather data, prepares input sequences, makes predictions,
    and returns daily averaged forecasts with beach safety recommendations.
    
    Args:
        latitude (float): Location latitude
        longitude (float): Location longitude
        
    Returns:
        List[Dict]: List of 3 daily forecast dictionaries
        
    Raises:
        ValueError: If data loading fails
        RuntimeError: If prediction fails
    """
    try:
        print(f"\n🌍 Predicting weather for location: ({latitude}, {longitude})")
        
        # Load and prepare data
        df, scalers = load_weather_data()
        
        # Prepare input sequence
        input_tensor = prepare_input_sequence(df)
        
        # Add location-based variation
        input_tensor = add_location_variation(input_tensor, latitude, longitude)
        
        # Make prediction
        prediction_np = make_prediction(input_tensor)
        
        # Denormalize predictions
        denorm_pred = denormalize_predictions(prediction_np, scalers)
        
        # Generate daily forecasts
        forecast_output = generate_daily_forecasts(denorm_pred)
        
        print(f"\n✅ Successfully generated 3-day forecast")
        return forecast_output
        
    except Exception as e:
        error_msg = f"Weather prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Validate geographic coordinates.
    
    Args:
        latitude (float): Latitude value
        longitude (float): Longitude value
        
    Returns:
        bool: True if coordinates are valid
    """
    return -90 <= latitude <= 90 and -180 <= longitude <= 180

def get_model_info() -> Dict:
    """
    Get information about the loaded model.
    
    Returns:
        Dict: Model information
    """
    return {
        "model_type": "LSTM",
        "input_features": INPUT_FEATURES,
        "target_features": TARGET_FEATURES,
        "input_hours": INPUT_HOURS,
        "output_hours": OUTPUT_HOURS,
        "device": str(DEVICE),
        "model_parameters": sum(p.numel() for p in MODEL.parameters())
    }

# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Command-line interface for weather prediction.
    
    Usage:
        python predict_weather.py
    """
    try:
        print("🌊 WaveFarer Weather Prediction CLI")
        print("=" * 40)
        
        # Get user input
        lat = float(input("Enter latitude: "))
        lon = float(input("Enter longitude: "))
        
        # Validate coordinates
        if not validate_coordinates(lat, lon):
            print("❌ Invalid coordinates provided")
            exit(1)
        
        # Make prediction
        forecast = predict_weather_forecast(lat, lon)
        
        print("\n🎉 Prediction completed successfully!")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)