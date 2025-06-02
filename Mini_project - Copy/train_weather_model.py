import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler

INPUT_HOURS = 24
OUTPUT_HOURS = 72
BATCH_SIZE = 32
EPOCHS = 20
HIDDEN_SIZE = 128
DATA_DIR = "./weather_data"
MODEL_SAVE_PATH = "./models/weather_forecast_model_v2.pth"  

input_features = ['temperature', 'humidity', 'wind_direction', 'wind_speed']
target_features = ['temperature', 'wind_speed', 'wind_direction', 'wave_height', 'sea_surface_temp'] 

def load_all_data(data_dir):
    dfs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_dir, fname), skiprows=3)
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
    full_df = pd.concat(dfs).sort_values(by="time").reset_index(drop=True)
    return full_df

df = load_all_data(DATA_DIR)

scalers = {col: MinMaxScaler() for col in df.columns if col != 'time'}
for col in scalers:
    df[col] = scalers[col].fit_transform(df[[col]])

class WeatherDataset(Dataset):
    def __init__(self, data, input_len, output_len):
        self.X, self.y = [], []
        for i in range(len(data) - input_len - output_len):
            x_seq = data[input_features].iloc[i:i+input_len].values
            y_seq = data[target_features].iloc[i+input_len:i+input_len+output_len].values
            self.X.append(x_seq)
            self.y.append(y_seq)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

dataset = WeatherDataset(df, INPUT_HOURS, OUTPUT_HOURS)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WeatherForecastModel(
    input_size=len(input_features),
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_HOURS * len(target_features)
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            val_loss += criterion(output, y_batch).item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")