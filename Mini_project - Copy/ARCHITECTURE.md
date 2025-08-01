# WaveFarer Architecture Documentation 🏗️

## Overview

WaveFarer uses a **microservices architecture** with two distinct backends, each handling specific responsibilities:

- **Flask (Python)**: ML/AI Services
- **Express (Node.js)**: User Management Services

## 🏗️ **Architecture Diagram**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │  Express API    │
│   (React)       │◄──►│   (Python)      │    │   (Node.js)     │
│                 │    │                 │    │                 │
│ - Map Interface │    │ - ML Predictions│    │ - Authentication│
│ - Weather Data  │    │ - Weather Model │    │ - User Profiles │
│ - Safety Alerts │    │ - Safety Alerts │    │ - Data Storage  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   MongoDB       │    │   MongoDB       │
                       │   (Predictions) │    │   (Users)       │
                       └─────────────────┘    └─────────────────┘
```

## 🔧 **Service Responsibilities**

### **Flask API (Python) - Port 5001**
**Purpose**: Machine Learning and Weather Services

**Endpoints**:
- `POST /api/predict` - Beach safety predictions
- `POST /api/future-predict` - 3-day weather forecasts
- `GET /api/health` - Health check

**Features**:
- ✅ ML model inference
- ✅ Weather data processing
- ✅ Safety recommendations
- ✅ Rate limiting
- ✅ Security headers

**Dependencies**:
- PyTorch (ML models)
- Scikit-learn
- Pandas
- Flask

### **Express API (Node.js) - Port 3000**
**Purpose**: User Management and Authentication

**Endpoints**:
- `POST /signup` - User registration
- `POST /login` - User authentication
- `PUT /update-profile/:email` - Profile updates
- `GET /get-user/:email` - User data retrieval
- `GET /api/fetchdata` - Prediction history

**Features**:
- ✅ User authentication (JWT)
- ✅ Password hashing
- ✅ Profile management
- ✅ Data persistence

**Dependencies**:
- Express.js
- Mongoose (MongoDB ODM)
- JWT
- bcryptjs

## 🚀 **Deployment Strategy**

### **Development Environment**
```bash
# Terminal 1: Flask API (ML Services)
cd WaveFarerFin/Mini_project\ -\ Copy/
python app.py
# Runs on http://localhost:5001

# Terminal 2: Express API (User Services)
cd WaveFarerFin/Mini_project\ -\ Copy/backend/
npm start
# Runs on http://localhost:3000

# Terminal 3: Frontend
cd WaveFarerFin/Mini_project\ -\ Copy/frontend/
npm run dev
# Runs on http://localhost:5173
```

### **Production Environment**
```bash
# Flask API (ML Services)
gunicorn app:app -b 0.0.0.0:5001

# Express API (User Services)
pm2 start ecosystem.config.js

# Frontend (Static files)
npm run build
```

## 📊 **Data Flow**

### **1. User Authentication Flow**
```
Frontend → Express API → MongoDB (Users)
     ↓
JWT Token → Frontend Storage
```

### **2. Prediction Flow**
```
Frontend → Flask API → ML Models → MongoDB (Predictions)
     ↓
Safety Recommendations → Frontend Display
```

### **3. Weather Forecast Flow**
```
Frontend → Flask API → Weather Model → Processed Data
     ↓
3-Day Forecast → Frontend Display
```

## 🔒 **Security Architecture**

### **Flask API Security**
- ✅ Rate limiting (100 req/hour)
- ✅ Input validation
- ✅ Security headers
- ✅ CORS configuration
- ✅ Environment variables

### **Express API Security**
- ✅ JWT authentication
- ✅ Password hashing (bcrypt)
- ✅ Input sanitization
- ✅ CORS configuration
- ✅ Environment variables

## 📁 **File Structure**

```
WaveFarerFin/
├── app.py                    # Flask API (ML Services)
├── predict_weather.py        # Weather forecasting
├── requirements.txt          # Python dependencies
├── backend/                  # Express API (User Services)
│   ├── index.js             # Main Express server
│   ├── package.json         # Node.js dependencies
│   └── models/              # MongoDB schemas
│       ├── User.js          # User model
│       └── Prediction.js    # Prediction model
├── frontend/                 # React application
│   ├── src/
│   └── package.json
└── models/                   # ML model files
    ├── weather_forecast_model.pth
    └── alert_model.pkl
```

## 🔄 **API Integration**

### **Frontend Configuration**
```javascript
// API endpoints configuration
const API_CONFIG = {
  ML_API: 'http://localhost:5001',    // Flask API
  USER_API: 'http://localhost:3000',  // Express API
  FRONTEND: 'http://localhost:5173'   // React app
};
```

### **Cross-Service Communication**
```javascript
// Example: Making prediction with user context
const makePrediction = async (userData, weatherData) => {
  // 1. Get user info from Express API
  const user = await fetch(`${USER_API}/get-user/${userData.email}`);
  
  // 2. Make prediction with Flask API
  const prediction = await fetch(`${ML_API}/api/predict`, {
    method: 'POST',
    body: JSON.stringify({ ...weatherData, user_mail: userData.email })
  });
  
  return prediction;
};
```

## 🎯 **Benefits of This Architecture**

### **1. Separation of Concerns**
- **ML Services**: Isolated from user management
- **User Services**: Focused on authentication and profiles
- **Frontend**: Clean integration with both services

### **2. Scalability**
- **Independent scaling**: Scale ML services separately from user services
- **Technology choice**: Use best tools for each domain
- **Deployment flexibility**: Deploy services independently

### **3. Maintainability**
- **Clear boundaries**: Each service has specific responsibilities
- **Team separation**: Different teams can work on different services
- **Testing**: Test each service independently

### **4. Technology Optimization**
- **Python**: Best for ML/AI workloads
- **Node.js**: Best for user management and real-time features
- **React**: Best for interactive frontend

## 🚨 **Migration Recommendations**

### **Option 1: Keep Current Architecture** ✅
**Pros**:
- Clear separation of concerns
- Technology-specific optimizations
- Independent scaling

**Cons**:
- More complex deployment
- Two separate codebases

### **Option 2: Consolidate to Flask** 
**Pros**:
- Single codebase
- Simpler deployment

**Cons**:
- Less optimal for user management
- Larger monolithic application

### **Option 3: Consolidate to Express**
**Pros**:
- Single codebase
- Better for real-time features

**Cons**:
- Less optimal for ML workloads
- Python ML libraries unavailable

## 📋 **Recommended Approach**

**Keep the current microservices architecture** because:

1. ✅ **Clear separation**: ML and user services are distinct domains
2. ✅ **Technology fit**: Python for ML, Node.js for user management
3. ✅ **Scalability**: Services can scale independently
4. ✅ **Maintainability**: Each service has focused responsibilities

## 🔧 **Next Steps**

1. **Document API contracts** between services
2. **Implement service discovery** for production
3. **Add monitoring** for both services
4. **Create deployment scripts** for both environments
5. **Add comprehensive testing** for each service

---

*This architecture provides the best of both worlds: optimal technology choices for each domain while maintaining clear separation of concerns.* 🎉 