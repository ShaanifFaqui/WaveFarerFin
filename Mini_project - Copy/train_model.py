import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("beach_conditions_dataset.csv")  

data.columns = [col.replace('Â', '') for col in data.columns]
data.dropna(subset=['Beach Name', 'Weather Condition', 'Alert Message', 'Safety Message'], inplace=True)
alert_to_safety = dict(zip(data['Alert Message'], data['Safety Message']))
with open('alert_to_safety_mapping.json', 'w') as f:
    json.dump(alert_to_safety, f)
categorical_cols = ['Beach Name', 'Weather Condition']
data = pd.get_dummies(data, columns=categorical_cols)
label_encoder = LabelEncoder()
data['Alert_Code'] = label_encoder.fit_transform(data['Alert Message'])
alert_mapping = {label: int(code) for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
with open('alert_mapping.json', 'w') as f:
    json.dump(alert_mapping, f)

X = data.drop(columns=['Alert Message', 'Alert_Code', 'Safety Message'])  
y = data['Alert_Code']
with open('model_columns.json', 'w') as f:
    json.dump(list(X.columns), f)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'alert_model.pkl')

print("Model trained and saved successfully.")
print("Label mappings, safety messages, and training metadata saved.")
