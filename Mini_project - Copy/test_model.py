import pandas as pd
import joblib
import json

model = joblib.load('alert_model.pkl')

with open('alert_mapping.json', 'r') as f:
    alert_mapping = json.load(f)

code_to_alert = {int(v): k for k, v in alert_mapping.items()}

with open('alert_to_safety_mapping.json', 'r') as f:
    alert_to_safety = json.load(f)

with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

input_data = {
    'Latitude': 13.0827,
    'Longitude': 80.2707,
    'Temperature (°C)': 29.5,
    'Humidity (%)': 78,
    'Wind Speed (m/s)': 5.2,
    'Cloud Cover (%)': 65,
    'Wave Height (m)': 1.2,
    'Ocean Current Velocity (m/s)': 0.9,
    'Sea Surface Temp (°C)': 30.1,
    'Weather Condition': 2,
    'Beach Name': 'Goa Beach'
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

print("Predicted Alert Message:", alert_message)
print("Safety Message:", safety_message)
