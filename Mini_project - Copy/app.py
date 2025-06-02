from flask import Flask, request, jsonify
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
from models.weather_forecast_model import WeatherForecastModel

from flask import Flask, request, jsonify
from predict_weather import predict_weather_forecast  

app = Flask(__name__)
CORS(app) 

client = MongoClient(os.environ.get("MONGODB_URI", "mongodb+srv://wavefarer:wave@cluster0.cvpv8tc.mongodb.net/"))
db = client['test']
collection = db['predictions']

model = joblib.load('alert_model.pkl')

with open('alert_mapping.json', 'r') as f:
    alert_mapping = json.load(f)

code_to_alert = {int(v): k for k, v in alert_mapping.items()}

with open('alert_to_safety_mapping.json', 'r') as f:
    alert_to_safety = json.load(f)

with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

INPUT_HOURS = 24
input_features = ['temperature', 'humidity', 'wind_direction', 'wind_speed']
target_features = ['temperature', 'wind_speed', 'wind_direction']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_data():
    DATA_DIR = "./weather_data"
    dfs = []
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
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            dfs.append(df)

    df = pd.concat(dfs).sort_values(by="time").reset_index(drop=True)

    scalers = {col: MinMaxScaler() for col in df.columns if col != 'time'}
    for col in scalers:
        df[col] = scalers[col].fit_transform(df[[col]])

    return df, scalers

df, scalers = load_and_prepare_data()
weather_model = WeatherForecastModel(input_size=4, hidden_size=128, output_size=72 * 3).to(device)
weather_model.load_state_dict(torch.load("models/weather_forecast_model.pth", map_location=device))
weather_model.eval()

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received data: {data}")

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
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]
        predicted_code = model.predict(input_df)[0]
        alert_message = code_to_alert[predicted_code]
        safety_message = alert_to_safety.get(alert_message, "No safety message available.")
        response = {
            'alert_message': alert_message,
            'safety_message': safety_message
        }
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

        return jsonify(response)

    except Exception as e:
        return jsonify({"message": str(e), "status": "error"})


@app.route('/api/future-predict', methods=['POST'])  
def forecast():
    try:
        data = request.get_json()
        if not data or 'lat' not in data or 'lon' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing latitude or longitude in request body."
            }), 400

        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        forecast_data = predict_weather_forecast(lat, lon)
        return jsonify({
            "status": "success",
            "latitude": lat,
            "longitude": lon,
            "forecast": forecast_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
