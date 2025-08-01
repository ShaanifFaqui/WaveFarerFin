# WaveFarer - Beach Safety Prediction System 🌊

A comprehensive beach safety prediction system that combines weather forecasting, marine conditions, and AI-powered safety alerts.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 14+
- MongoDB (local or cloud)

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your MongoDB URI

# Start the Flask server
python app.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend (Node.js) Setup
```bash
cd backend
npm install
npm start
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:
```
MONGODB_URI=mongodb://localhost:27017/
FLASK_ENV=development
FLASK_DEBUG=True
```

## 🐛 Troubleshooting

### Common Issues:
1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **MongoDB Connection**: Ensure MongoDB is running or update MONGODB_URI
3. **Port Conflicts**: Change port in app.py if 5001 is busy

## 📁 Project Structure
```
WaveFarerFin/
├── app.py                 # Main Flask application
├── predict_weather.py     # Weather forecasting logic
├── requirements.txt       # Python dependencies
├── frontend/             # React application
├── backend/              # Node.js backend
└── models/               # ML models
```

## 🛠️ Development

### Running Tests
```bash
python test_model.py
```

### Model Training
```bash
python train_weather_model.py
```

## 📊 API Endpoints

- `POST /api/predict` - Get safety prediction
- `POST /api/future-predict` - Get 3-day forecast

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License 