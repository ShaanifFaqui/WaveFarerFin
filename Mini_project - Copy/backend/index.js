const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const axios = require('axios');

// Import middleware
const { 
  errorHandler, 
  asyncHandler, 
  notFoundHandler,
  validateSignup,
  validateLogin,
  validateProfileUpdate,
  validatePredictionFetch,
  authenticateToken,
  authorizeUser,
  authLimiter,
  apiLimiter
} = require('./middleware/errorHandler');

const { 
  logger, 
  requestLogger, 
  errorLogger, 
  performanceMonitor, 
  securityLogger 
} = require('./middleware/logger');

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

// =============================================================================
// MIDDLEWARE SETUP
// =============================================================================

// Security and monitoring middleware
app.use(securityLogger);
app.use(performanceMonitor);
app.use(requestLogger);

// Rate limiting
app.use('/api/auth', authLimiter);
app.use('/api', apiLimiter);

// Standard middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));


// MongoDB connection
mongoose.connect(process.env.MONGODB_URL)
.then(() => console.log("MongoDB connected"))
.catch(err => console.error(err));

//  model
const User = require('./models/User');
const Prediction = require('./models/Prediction'); 

// =============================================================================
// AUTHENTICATION ROUTES
// =============================================================================

// =============================================================================
// AUTHENTICATION ROUTES
// =============================================================================

// Signup Route
app.post('/api/auth/signup', validateSignup, asyncHandler(async (req, res) => {
  const { name, email, password } = req.body;
  
  const existingUser = await User.findOne({ email });
  if (existingUser) {
    throw new ConflictError('User already exists');
  }

  const hashedPassword = await bcrypt.hash(password, 10);
  const user = new User({ name, email, password: hashedPassword });
  await user.save();
  
  logger.logAuth('signup', email, true);
  
  res.status(201).json({ 
    success: true,
    message: 'Signup successful',
    user: { name: user.name, email: user.email }
  });
}));

// Login Route
app.post('/api/auth/login', validateLogin, asyncHandler(async (req, res) => {
  const { email, password } = req.body;
  
  const user = await User.findOne({ email });
  if (!user) {
    logger.logAuth('login', email, false);
    throw new AuthenticationError('Invalid credentials');
  }

  const isMatch = await bcrypt.compare(password, user.password);
  if (!isMatch) {
    logger.logAuth('login', email, false);
    throw new AuthenticationError('Invalid credentials');
  }

  // Create JWT token
  const token = jwt.sign({ id: user._id, email: user.email }, process.env.JWT_SECRET, {
    expiresIn: '24h',
  });

  logger.logAuth('login', email, true);
  
  res.status(200).json({
    success: true,
    message: 'Login successful',
    token,
    user: { name: user.name, email: user.email }
  });
}));

// =============================================================================
// USER PROFILE ROUTES
// =============================================================================

// =============================================================================
// USER PROFILE ROUTES
// =============================================================================

app.put('/api/users/:email', authenticateToken, authorizeUser, validateProfileUpdate, asyncHandler(async (req, res) => {
  const { email } = req.params;
  
  const updatedUser = await User.findOneAndUpdate(
    { email },
    {
      mobile: req.body.phone,
      location: req.body.location,
      preferedBeach: req.body.preferredBeaches,
      Language: req.body.language,
    },
    { new: true }
  );
  
  if (!updatedUser) {
    throw new NotFoundError('User');
  }
  
  logger.info('User profile updated', { email });
  
  res.json({
    success: true,
    message: 'Profile updated successfully',
    user: updatedUser
  });
}));

app.get('/api/users/:email', authenticateToken, authorizeUser, asyncHandler(async (req, res) => {
  const { email } = req.params;
  
  const user = await User.findOne({ email });
  if (!user) {
    throw new NotFoundError('User');
  }
  
  res.status(200).json({
    success: true,
    user
  });
}));

// =============================================================================
// PREDICTION HISTORY ROUTES
// =============================================================================

// =============================================================================
// PREDICTION HISTORY ROUTES
// =============================================================================

// Get user's prediction history
app.get('/api/predictions/:email', authenticateToken, authorizeUser, asyncHandler(async (req, res) => {
  const { email } = req.params;
  
  const predictions = await Prediction.find({ email })
    .sort({ timestamp: -1 })
    .limit(10); // Get last 10 predictions

  logger.logPrediction('fetch_history', 'multiple', email, true);

  res.json({
    success: true,
    email,
    predictions: predictions.map(pred => ({
      timestamp: pred.timestamp,
      BeachName: pred.BeachName,
      alert_message: pred.prediction?.alert_message,
      safety_message: pred.prediction?.safety_message
    }))
  });
}));

// Get specific prediction data
app.get('/api/predictions/fetch', validatePredictionFetch, asyncHandler(async (req, res) => {
  const { userEmail, beachName } = req.query;
  const BeachName = decodeURIComponent(beachName).replace(/\+/g, ' ');

  // Calculate ±2 minutes time range
  const now = new Date();
  const startTime = new Date(now.getTime() - 2 * 60 * 1000); // 2 minutes ago
  const endTime = new Date(now.getTime() + 2 * 60 * 1000);   // 2 minutes 

  const data = await Prediction.findOne({
    BeachName: BeachName,
    email: userEmail,
    timestamp: { $gte: startTime, $lte: endTime }
  }).sort({ timestamp: -1 }); // in case there are multiple, get the latest

  if (!data) {
    throw new NotFoundError('Recent prediction');
  }

  logger.logPrediction('fetch_specific', BeachName, userEmail, true);

  res.json({
    success: true,
    data
  });
}));

// =============================================================================
// HEALTH CHECK ROUTE
// =============================================================================

app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    status: 'healthy',
    message: 'WaveFarer User API is running',
    timestamp: new Date().toISOString(),
    service: 'user-management',
    version: '1.0.0'
  });
});

// =============================================================================
// ERROR HANDLING MIDDLEWARE
// =============================================================================

// 404 handler - must be after all routes
app.use(notFoundHandler);

// Global error handler - must be last
app.use(errorHandler);

// =============================================================================
// SERVER STARTUP
// =============================================================================

// MongoDB connection with error handling
mongoose.connect(process.env.MONGODB_URL)
  .then(() => {
    logger.logSystem('MongoDB connected successfully');
    
    app.listen(PORT, () => {
      logger.logSystem('Server started', {
        port: PORT,
        environment: process.env.NODE_ENV || 'development',
        mongodbStatus: mongoose.connection.readyState === 1 ? 'Connected' : 'Disconnected'
      });
      
      console.log(`🌊 WaveFarer User API running on port ${PORT}`);
      console.log(`📊 MongoDB: ${mongoose.connection.readyState === 1 ? 'Connected' : 'Disconnected'}`);
      console.log(`🔧 Environment: ${process.env.NODE_ENV || 'development'}`);
    });
  })
  .catch((error) => {
    logger.error('MongoDB connection failed', { error: error.message });
    console.error('❌ MongoDB connection failed:', error.message);
    process.exit(1);
  });

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.logSystem('SIGTERM received, shutting down gracefully');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.logSystem('SIGINT received, shutting down gracefully');
  process.exit(0);
});
