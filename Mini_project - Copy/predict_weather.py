import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# === Configuration ===
input_features = ['temperature', 'humidity', 'wind_direction', 'wind_speed']
target_features = ['temperature', 'wind_speed', 'wind_direction', 'wave_height', 'sea_surface_temp']
INPUT_HOURS = 24
OUTPUT_HOURS = 72
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Define the Model ===
class WeatherForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(WeatherForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output.view(-1, OUTPUT_HOURS, len(target_features))

# === Load Model ===
model = WeatherForecastModel(
    input_size=len(input_features),
    hidden_size=128,
    output_size=OUTPUT_HOURS * len(target_features)
).to(device)

model.load_state_dict(torch.load("models/weather_forecast_model_v2.pth", map_location=device))
model.eval()

def predict_weather_forecast(latitude: float, longitude: float):
    # === Load and Preprocess Data ===
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

    # === Normalize ===
    scalers = {col: MinMaxScaler() for col in df.columns if col != 'time'}
    for col in scalers:
        df[col] = scalers[col].fit_transform(df[[col]])

    # === Prepare Input ===
    input_seq = df[input_features].iloc[-INPUT_HOURS:].values
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

    # === Add Location-Based Randomness ===
    seed = int((latitude + 90) * 1000 + (longitude + 180))
    np.random.seed(seed)
    noise = np.random.normal(0, 0.02, input_tensor.shape)
    input_tensor += torch.tensor(noise, dtype=torch.float32).to(device)

    # === Predict ===
    with torch.no_grad():
        prediction = model(input_tensor)

    # === Denormalize ===
    prediction_np = prediction.squeeze(0).cpu().numpy()
    denorm_pred = np.zeros_like(prediction_np)
    for i, feature in enumerate(target_features):
        denorm_pred[:, i] = scalers[feature].inverse_transform(prediction_np[:, i].reshape(-1, 1)).flatten()

    # === Display Forecast ===
    print("\n📆 3-Day Weather Forecast (Averaged per Day):")
    forecast_output = []
    for day in range(3):
        day_data = denorm_pred[day * 24:(day + 1) * 24]
        avg_temp = float(np.mean(day_data[:, 0]))
        avg_wind_spd = float(np.mean(day_data[:, 1]))
        avg_wind_dir = float(np.mean(day_data[:, 2]))
        avg_wave = float(np.mean(day_data[:, 3]))
        avg_sea_temp = float(np.mean(day_data[:, 4]))

        if avg_wave < 2 and avg_wind_spd < 15:
            beach_msg = "✅ Safe to go to the beach!"
        else:
            beach_msg = "⚠ Not suitable for beach activities."

        print(f"\nDay {day+1}:")
        print(f"Avg Temp = {avg_temp:.2f}°C")
        print(f"Avg Wind Speed = {avg_wind_spd:.2f} m/s, Direction = {avg_wind_dir:.2f}°")
        print(f"Avg Wave Height = {avg_wave:.2f} m")
        print(f"Sea Surface Temp = {avg_sea_temp:.2f}°C")
        print(f"Beach Recommendation: {beach_msg}")

        forecast_output.append({
            "day": day + 1,
            "avg_temp": avg_temp,
            "avg_wind_speed": avg_wind_spd,
            "avg_wind_direction": avg_wind_dir,
            "avg_wave_height": avg_wave,
            "avg_sea_surface_temp": avg_sea_temp,
            "beach_safety": beach_msg
        })

    return forecast_output

# === Optional CLI ===
if __name__ == "_main_":
    lat = float(input("Enter latitude: "))
    lon = float(input("Enter longitude: "))
    predict_weather_forecast(lat, lon)