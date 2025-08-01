# WaveFarer API Endpoints Documentation 🔌

## Overview

This document standardizes all API endpoints and port usage across the WaveFarer microservices architecture.

## 🏗️ **Service Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │  Express API    │
│   (React)       │◄──►│   (Python)      │    │   (Node.js)     │
│   Port: 5173    │    │   Port: 5001    │    │   Port: 3000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 **Standardized API Endpoints**

### **Flask API (ML Services) - Port 5001**
**Base URL**: `http://localhost:5001`

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/api/predict` | Beach safety prediction | Weather data | Safety alert |
| `POST` | `/api/future-predict` | 3-day weather forecast | Coordinates | Forecast data |
| `GET` | `/api/health` | Health check | None | Status info |

### **Express API (User Services) - Port 3000**
**Base URL**: `http://localhost:3000`

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/api/auth/signup` | User registration | User data | Success message |
| `POST` | `/api/auth/login` | User authentication | Credentials | JWT token |
| `GET` | `/api/users/:email` | Get user profile | None | User data |
| `PUT` | `/api/users/:email` | Update user profile | Profile data | Updated user |
| `GET` | `/api/predictions/:email` | Get prediction history | None | Predictions |

## 🔧 **API Configuration**

### **Frontend Configuration**
```javascript
// src/config/api.js
const API_CONFIG = {
  // ML Services (Flask)
  ML_API: {
    baseURL: process.env.REACT_APP_ML_API_URL || 'http://localhost:5001',
    endpoints: {
      predict: '/api/predict',
      futurePredict: '/api/future-predict',
      health: '/api/health'
    }
  },
  
  // User Services (Express)
  USER_API: {
    baseURL: process.env.REACT_APP_USER_API_URL || 'http://localhost:3000',
    endpoints: {
      signup: '/api/auth/signup',
      login: '/api/auth/login',
      getUser: (email) => `/api/users/${email}`,
      updateUser: (email) => `/api/users/${email}`,
      getPredictions: (email) => `/api/predictions/${email}`
    }
  }
};

export default API_CONFIG;
```

### **Environment Variables**
```bash
# Frontend (.env)
REACT_APP_ML_API_URL=http://localhost:5001
REACT_APP_USER_API_URL=http://localhost:3000

# Flask API (.env)
FLASK_PORT=5001
FLASK_HOST=0.0.0.0

# Express API (.env)
EXPRESS_PORT=3000
EXPRESS_HOST=0.0.0.0
```

## 📊 **Detailed Endpoint Specifications**

### **1. Flask API Endpoints**

#### **POST /api/predict**
**Purpose**: Get beach safety prediction based on current conditions

**Request Body**:
```json
{
  "latitude": 13.0827,
  "longitude": 80.2707,
  "Temperature": 28.5,
  "Humidity": 75,
  "WindSpeed": 12.3,
  "CloudCover": 45,
  "WeatherCode": 1,
  "WaveHeight": 1.2,
  "OceanCurrentVelocity": 0.3,
  "SeaSurfaceTemp": 26.8,
  "BeachName": "Marina Beach",
  "user_mail": "user@example.com"
}
```

**Response**:
```json
{
  "alert_message": "Moderate Risk",
  "safety_message": "Exercise caution. Check local conditions."
}
```

#### **POST /api/future-predict**
**Purpose**: Get 3-day weather forecast

**Request Body**:
```json
{
  "lat": 13.0827,
  "lon": 80.2707
}
```

**Response**:
```json
{
  "status": "success",
  "latitude": 13.0827,
  "longitude": 80.2707,
  "forecast": [
    {
      "day": 1,
      "avg_temp": 28.5,
      "avg_wind_speed": 12.3,
      "avg_wind_direction": 180.2,
      "avg_wave_height": 1.2,
      "avg_sea_surface_temp": 26.8,
      "beach_safety": "✅ Safe to go to the beach!"
    }
  ]
}
```

#### **GET /api/health**
**Purpose**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "message": "WaveFarer ML API is running",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **2. Express API Endpoints**

#### **POST /api/auth/signup**
**Purpose**: User registration

**Request Body**:
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response**:
```json
{
  "message": "Signup successful",
  "user": {
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

#### **POST /api/auth/login**
**Purpose**: User authentication

**Request Body**:
```json
{
  "email": "john@example.com",
  "password": "securepassword123"
}
```

**Response**:
```json
{
  "message": "Login successful",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

#### **GET /api/users/:email**
**Purpose**: Get user profile

**Response**:
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "mobile": "+1234567890",
  "location": "Chennai, India",
  "preferedBeach": ["Marina Beach", "Elliot's Beach"],
  "Language": "English"
}
```

#### **PUT /api/users/:email**
**Purpose**: Update user profile

**Request Body**:
```json
{
  "phone": "+1234567890",
  "location": "Chennai, India",
  "preferredBeaches": ["Marina Beach", "Elliot's Beach"],
  "language": "English"
}
```

**Response**:
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "mobile": "+1234567890",
  "location": "Chennai, India",
  "preferedBeach": ["Marina Beach", "Elliot's Beach"],
  "Language": "English"
}
```

#### **GET /api/predictions/:email**
**Purpose**: Get user's prediction history

**Response**:
```json
{
  "email": "john@example.com",
  "predictions": [
    {
      "timestamp": "2024-01-15T10:30:00Z",
      "BeachName": "Marina Beach",
      "alert_message": "Moderate Risk",
      "safety_message": "Exercise caution. Check local conditions."
    }
  ]
}
```

## 🔄 **Frontend Integration**

### **API Service Functions**
```javascript
// src/services/api.js
import API_CONFIG from '../config/api';

class APIService {
  // ML API Methods
  async makePrediction(weatherData) {
    const response = await fetch(`${API_CONFIG.ML_API.baseURL}${API_CONFIG.ML_API.endpoints.predict}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(weatherData)
    });
    return response.json();
  }

  async getForecast(lat, lon) {
    const response = await fetch(`${API_CONFIG.ML_API.baseURL}${API_CONFIG.ML_API.endpoints.futurePredict}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon })
    });
    return response.json();
  }

  // User API Methods
  async signup(userData) {
    const response = await fetch(`${API_CONFIG.USER_API.baseURL}${API_CONFIG.USER_API.endpoints.signup}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });
    return response.json();
  }

  async login(credentials) {
    const response = await fetch(`${API_CONFIG.USER_API.baseURL}${API_CONFIG.USER_API.endpoints.login}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials)
    });
    return response.json();
  }

  async getUserProfile(email, token) {
    const response = await fetch(`${API_CONFIG.USER_API.baseURL}${API_CONFIG.USER_API.endpoints.getUser(email)}`, {
      method: 'GET',
      headers: { 
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
    return response.json();
  }
}

export default new APIService();
```

## 🚨 **Error Handling**

### **Standard Error Response Format**
```json
{
  "error": "Error message",
  "status": "error",
  "code": 400
}
```

### **HTTP Status Codes**
- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `404`: Not Found
- `429`: Rate Limit Exceeded
- `500`: Internal Server Error

## 🔒 **Security Headers**

All API responses include security headers:
```javascript
{
  'X-Content-Type-Options': 'nosniff',
  'X-Frame-Options': 'DENY',
  'X-XSS-Protection': '1; mode=block',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}
```

## 📋 **Testing Endpoints**

### **Health Check**
```bash
# Flask API
curl http://localhost:5001/api/health

# Express API
curl http://localhost:3000/api/health
```

### **Test Prediction**
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 13.0827,
    "longitude": 80.2707,
    "Temperature": 28.5,
    "Humidity": 75,
    "WindSpeed": 12.3,
    "CloudCover": 45,
    "WeatherCode": 1,
    "WaveHeight": 1.2,
    "OceanCurrentVelocity": 0.3,
    "SeaSurfaceTemp": 26.8,
    "BeachName": "Marina Beach",
    "user_mail": "test@example.com"
  }'
```

## 🎯 **Benefits of Standardization**

1. ✅ **Consistent naming**: All endpoints follow REST conventions
2. ✅ **Clear separation**: ML vs User services clearly defined
3. ✅ **Easy integration**: Frontend can easily switch between environments
4. ✅ **Documentation**: Complete API documentation
5. ✅ **Testing**: Standardized testing procedures
6. ✅ **Security**: Consistent security headers and validation

---

*This standardization ensures consistent API usage across all services and makes the system more maintainable and scalable.* 🚀 