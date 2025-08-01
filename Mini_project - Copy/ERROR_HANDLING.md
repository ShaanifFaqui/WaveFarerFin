# WaveFarer Error Handling System 🛡️

## Overview

This document describes the comprehensive error handling system implemented across the WaveFarer microservices architecture, ensuring robust error management, user-friendly messages, and proper logging.

## 🏗️ **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Flask API     │    │  Express API    │
│   (React)       │◄──►│   (Python)      │    │   (Node.js)     │
│   Error Handler │    │   Error Handler │    │   Error Handler │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 **Backend Error Handling (Express)**

### **Error Types**

| Error Class | Status Code | Description | Recoverable |
|-------------|-------------|-------------|-------------|
| `ValidationError` | 400 | Input validation failed | ✅ Yes |
| `AuthenticationError` | 401 | Authentication failed | ✅ Yes |
| `AuthorizationError` | 403 | Access denied | ❌ No |
| `NotFoundError` | 404 | Resource not found | ❌ No |
| `ConflictError` | 409 | Resource conflict | ✅ Yes |
| `RateLimitError` | 429 | Too many requests | ✅ Yes |
| `AppError` | 500 | Internal server error | ❌ No |

### **Middleware Stack**

```javascript
// Security and monitoring
app.use(securityLogger);
app.use(performanceMonitor);
app.use(requestLogger);

// Rate limiting
app.use('/api/auth', authLimiter);
app.use('/api', apiLimiter);

// Request validation
app.post('/api/auth/signup', validateSignup, asyncHandler(async (req, res) => {
  // Route logic
}));

// Error handling (last)
app.use(notFoundHandler);
app.use(errorHandler);
```

### **Validation Functions**

```javascript
// Email validation
validateEmail('user@example.com');

// Password validation
validatePassword('secure123');

// Required fields validation
validateRequired(req.body, ['name', 'email', 'password']);

// Coordinates validation
validateCoordinates(lat, lon);
```

### **Error Response Format**

```json
{
  "success": false,
  "error": {
    "message": "User already exists",
    "statusCode": 409,
    "name": "ConflictError"
  }
}
```

## 🎨 **Frontend Error Handling (React)**

### **Error Types**

| Error Class | Type | Description | User Action |
|-------------|------|-------------|-------------|
| `NetworkError` | NETWORK | Connection failed | Retry |
| `APIError` | API | Server error | Retry/Wait |
| `ValidationError` | VALIDATION | Invalid input | Fix |
| `AuthenticationError` | AUTH | Login required | Login |
| `UserInputError` | USER_INPUT | Invalid user input | Correct |

### **Error Handler Features**

```javascript
// Log error with context
errorHandler.logError(error, { context: 'prediction_api', payload });

// Show user-friendly error notification
errorHandler.showError(error);

// Show success notification
errorHandler.showSuccess('Operation completed successfully');

// Validate user input
errorHandler.validateEmail(email);
errorHandler.validatePassword(password);
errorHandler.validateCoordinates(lat, lon);
```

### **Error Notification System**

```javascript
// Automatic error notifications
errorHandler.showError(error, 5000); // 5 second duration

// Success notifications
errorHandler.showSuccess('Profile updated successfully', 3000);
```

## 🔧 **Implementation Examples**

### **Backend Route with Error Handling**

```javascript
app.post('/api/auth/signup', validateSignup, asyncHandler(async (req, res) => {
  const { name, email, password } = req.body;
  
  // Check if user exists
  const existingUser = await User.findOne({ email });
  if (existingUser) {
    throw new ConflictError('User already exists');
  }

  // Create user
  const hashedPassword = await bcrypt.hash(password, 10);
  const user = new User({ name, email, password: hashedPassword });
  await user.save();
  
  // Log success
  logger.logAuth('signup', email, true);
  
  res.status(201).json({ 
    success: true,
    message: 'Signup successful',
    user: { name: user.name, email: user.email }
  });
}));
```

### **Frontend API Call with Error Handling**

```javascript
const handlePrediction = async (weatherData) => {
  try {
    const prediction = await apiService.makePrediction(weatherData);
    errorHandler.showSuccess('Prediction completed successfully');
    return prediction;
  } catch (error) {
    errorHandler.logError(error, { context: 'prediction', weatherData });
    errorHandler.showError(error);
    throw error;
  }
};
```

## 📊 **Logging System**

### **Backend Logging**

```javascript
// Request logging
logger.info('Request started', {
  method: req.method,
  url: req.url,
  ip: req.ip,
  userAgent: req.get('User-Agent')
});

// Error logging
logger.error('Error occurred', {
  message: err.message,
  stack: err.stack,
  url: req.url,
  method: req.method
});

// Authentication logging
logger.logAuth('login', email, true);

// Prediction logging
logger.logPrediction('fetch_history', 'multiple', email, true);
```

### **Frontend Logging**

```javascript
// Error logging with context
errorHandler.logError(error, {
  context: 'prediction_api',
  payload: weatherData,
  userAgent: navigator.userAgent,
  url: window.location.href
});

// Get error log for debugging
const errorLog = errorHandler.getErrorLog();
```

## 🚨 **Security Features**

### **Rate Limiting**

```javascript
// Authentication rate limiting
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 requests per window
  message: 'Too many authentication attempts'
});

// API rate limiting
const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
  message: 'Too many requests'
});
```

### **Security Monitoring**

```javascript
// Suspicious activity detection
const suspiciousPatterns = [
  /\.\./, // Directory traversal
  /<script/i, // XSS attempts
  /union.*select/i, // SQL injection attempts
  /eval\(/i, // Code injection attempts
];

// Log security events
logger.logSecurity('suspicious_request', {
  url: req.url,
  userAgent: req.get('User-Agent'),
  ip: req.ip
});
```

## 🎯 **User Experience**

### **Error Messages**

| Error Type | User Message | Action |
|------------|--------------|--------|
| Network Error | "Unable to connect to the server. Please check your internet connection." | Retry |
| Authentication Error | "Please log in again to continue." | Login |
| Validation Error | "Please check your input and try again." | Fix |
| Rate Limit Error | "Please wait a moment before trying again." | Wait |
| Server Error | "The service is temporarily unavailable. Please try again later." | Retry |

### **Success Messages**

```javascript
// Automatic success notifications
errorHandler.showSuccess('Profile updated successfully');
errorHandler.showSuccess('Safety prediction completed');
errorHandler.showSuccess('Login successful');
```

## 🔍 **Debugging Features**

### **Development Mode**

```javascript
// Detailed error information in development
if (process.env.NODE_ENV === 'development') {
  res.status(statusCode).json({
    success: false,
    error: {
      message,
      statusCode,
      stack: err.stack,
      name: err.name
    }
  });
}
```

### **Error Log Access**

```javascript
// Get error log for debugging
const errorLog = errorHandler.getErrorLog();

// Clear error log
errorHandler.clearErrorLog();
```

## 📈 **Performance Monitoring**

### **Request Performance**

```javascript
// Monitor request duration
const start = process.hrtime();
res.on('finish', () => {
  const [seconds, nanoseconds] = process.hrtime(start);
  const duration = (seconds * 1000) + (nanoseconds / 1000000);
  
  logger.logPerformance('request_duration', duration.toFixed(2));
  
  // Log slow requests
  if (duration > 1000) {
    logger.warn('Slow request detected', {
      url: req.url,
      method: req.method,
      duration: `${duration.toFixed(2)}ms`
    });
  }
});
```

## 🛠️ **Configuration**

### **Environment Variables**

```bash
# Backend
LOG_LEVEL=INFO
LOG_FILE=wavefarer.log
NODE_ENV=development

# Frontend
REACT_APP_LOG_LEVEL=INFO
NODE_ENV=development
```

### **Error Handler Configuration**

```javascript
// Backend logger configuration
const logger = new Logger({
  logLevel: process.env.LOG_LEVEL || 'INFO',
  logFile: process.env.LOG_FILE || 'wavefarer.log'
});

// Frontend error handler configuration
const errorHandler = new ErrorHandler({
  maxLogSize: 100,
  isDevelopment: process.env.NODE_ENV === 'development'
});
```

## 🎯 **Benefits**

### **1. User Experience** ✨
- ✅ User-friendly error messages
- ✅ Automatic error notifications
- ✅ Success confirmations
- ✅ Clear action guidance

### **2. Developer Experience** 👨‍💻
- ✅ Comprehensive error logging
- ✅ Detailed error context
- ✅ Easy debugging tools
- ✅ Development vs production modes

### **3. System Reliability** 🛡️
- ✅ Graceful error recovery
- ✅ Rate limiting protection
- ✅ Security monitoring
- ✅ Performance tracking

### **4. Maintainability** 🔧
- ✅ Centralized error handling
- ✅ Consistent error formats
- ✅ Easy error tracking
- ✅ Scalable architecture

---

*This comprehensive error handling system ensures robust, user-friendly, and maintainable error management across the entire WaveFarer application.* 🚀 