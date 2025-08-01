/**
 * WaveFarer Error Handling Middleware
 * ====================================
 * 
 * Comprehensive error handling for the Express backend
 * with proper logging, validation, and user-friendly responses.
 */

const logger = require('./logger');

// =============================================================================
// ERROR TYPES
// =============================================================================

class AppError extends Error {
  constructor(message, statusCode, isOperational = true) {
    super(message);
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.status = `${statusCode}`.startsWith('4') ? 'fail' : 'error';
    
    Error.captureStackTrace(this, this.constructor);
  }
}

class ValidationError extends AppError {
  constructor(message) {
    super(message, 400);
    this.name = 'ValidationError';
  }
}

class AuthenticationError extends AppError {
  constructor(message = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
  }
}

class AuthorizationError extends AppError {
  constructor(message = 'Access denied') {
    super(message, 403);
    this.name = 'AuthorizationError';
  }
}

class NotFoundError extends AppError {
  constructor(resource = 'Resource') {
    super(`${resource} not found`, 404);
    this.name = 'NotFoundError';
  }
}

class ConflictError extends AppError {
  constructor(message = 'Resource conflict') {
    super(message, 409);
    this.name = 'ConflictError';
  }
}

class RateLimitError extends AppError {
  constructor(message = 'Too many requests') {
    super(message, 429);
    this.name = 'RateLimitError';
  }
}

// =============================================================================
// VALIDATION FUNCTIONS
// =============================================================================

/**
 * Validate email format
 */
function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!email || !emailRegex.test(email)) {
    throw new ValidationError('Invalid email format');
  }
  return email.toLowerCase().trim();
}

/**
 * Validate password strength
 */
function validatePassword(password) {
  if (!password || password.length < 6) {
    throw new ValidationError('Password must be at least 6 characters long');
  }
  return password;
}

/**
 * Validate required fields
 */
function validateRequired(data, fields) {
  const missing = [];
  fields.forEach(field => {
    if (!data[field] || (typeof data[field] === 'string' && data[field].trim() === '')) {
      missing.push(field);
    }
  });
  
  if (missing.length > 0) {
    throw new ValidationError(`Missing required fields: ${missing.join(', ')}`);
  }
}

/**
 * Validate coordinates
 */
function validateCoordinates(lat, lon) {
  if (lat === undefined || lon === undefined) {
    throw new ValidationError('Latitude and longitude are required');
  }
  
  const latNum = parseFloat(lat);
  const lonNum = parseFloat(lon);
  
  if (isNaN(latNum) || isNaN(lonNum)) {
    throw new ValidationError('Invalid coordinates format');
  }
  
  if (latNum < -90 || latNum > 90) {
    throw new ValidationError('Latitude must be between -90 and 90');
  }
  
  if (lonNum < -180 || lonNum > 180) {
    throw new ValidationError('Longitude must be between -180 and 180');
  }
  
  return { lat: latNum, lon: lonNum };
}

// =============================================================================
// ERROR HANDLING MIDDLEWARE
// =============================================================================

/**
 * Global error handler middleware
 */
function errorHandler(err, req, res, next) {
  let error = { ...err };
  error.message = err.message;

  // Log error
  logger.error('Error occurred:', {
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  // Mongoose validation error
  if (err.name === 'ValidationError') {
    const message = Object.values(err.errors).map(val => val.message).join(', ');
    error = new ValidationError(message);
  }

  // Mongoose duplicate key error
  if (err.code === 11000) {
    const field = Object.keys(err.keyValue)[0];
    error = new ConflictError(`${field} already exists`);
  }

  // Mongoose cast error (invalid ObjectId)
  if (err.name === 'CastError') {
    error = new ValidationError('Invalid ID format');
  }

  // JWT errors
  if (err.name === 'JsonWebTokenError') {
    error = new AuthenticationError('Invalid token');
  }

  if (err.name === 'TokenExpiredError') {
    error = new AuthenticationError('Token expired');
  }

  // Rate limiting error
  if (err.name === 'RateLimitError') {
    error = new RateLimitError(err.message);
  }

  // Default error
  const statusCode = error.statusCode || 500;
  const message = error.message || 'Internal server error';

  // Development error response
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
  } else {
    // Production error response
    res.status(statusCode).json({
      success: false,
      error: {
        message: statusCode === 500 ? 'Internal server error' : message,
        statusCode
      }
    });
  }
}

/**
 * Async error wrapper
 */
function asyncHandler(fn) {
  return (req, res, next) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

/**
 * 404 handler
 */
function notFoundHandler(req, res, next) {
  const error = new NotFoundError(`Route ${req.originalUrl}`);
  next(error);
}

// =============================================================================
// REQUEST VALIDATION MIDDLEWARE
// =============================================================================

/**
 * Validate signup request
 */
function validateSignup(req, res, next) {
  try {
    const { name, email, password } = req.body;
    
    validateRequired(req.body, ['name', 'email', 'password']);
    validateEmail(email);
    validatePassword(password);
    
    if (name.length < 2) {
      throw new ValidationError('Name must be at least 2 characters long');
    }
    
    next();
  } catch (error) {
    next(error);
  }
}

/**
 * Validate login request
 */
function validateLogin(req, res, next) {
  try {
    const { email, password } = req.body;
    
    validateRequired(req.body, ['email', 'password']);
    validateEmail(email);
    
    next();
  } catch (error) {
    next(error);
  }
}

/**
 * Validate user profile update
 */
function validateProfileUpdate(req, res, next) {
  try {
    const { phone, location, preferredBeaches, language } = req.body;
    
    if (phone && !/^\+?[\d\s\-\(\)]+$/.test(phone)) {
      throw new ValidationError('Invalid phone number format');
    }
    
    if (location && location.length < 2) {
      throw new ValidationError('Location must be at least 2 characters long');
    }
    
    if (preferredBeaches && !Array.isArray(preferredBeaches)) {
      throw new ValidationError('Preferred beaches must be an array');
    }
    
    next();
  } catch (error) {
    next(error);
  }
}

/**
 * Validate prediction fetch request
 */
function validatePredictionFetch(req, res, next) {
  try {
    const { userEmail, beachName } = req.query;
    
    validateRequired(req.query, ['userEmail', 'beachName']);
    validateEmail(userEmail);
    
    if (beachName.length < 2) {
      throw new ValidationError('Beach name must be at least 2 characters long');
    }
    
    next();
  } catch (error) {
    next(error);
  }
}

// =============================================================================
// AUTHENTICATION MIDDLEWARE
// =============================================================================

/**
 * Verify JWT token
 */
function authenticateToken(req, res, next) {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    
    if (!token) {
      throw new AuthenticationError('Access token required');
    }
    
    jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
      if (err) {
        if (err.name === 'TokenExpiredError') {
          throw new AuthenticationError('Token expired');
        }
        throw new AuthenticationError('Invalid token');
      }
      
      req.user = user;
      next();
    });
  } catch (error) {
    next(error);
  }
}

/**
 * Check if user owns the resource
 */
function authorizeUser(req, res, next) {
  try {
    const { email } = req.params;
    const userEmail = req.user.email;
    
    if (email !== userEmail) {
      throw new AuthorizationError('Access denied to this resource');
    }
    
    next();
  } catch (error) {
    next(error);
  }
}

// =============================================================================
// RATE LIMITING
// =============================================================================

const rateLimit = require('express-rate-limit');

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5, // 5 requests per window
  message: 'Too many authentication attempts, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

const apiLimiter = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 100, // 100 requests per minute
  message: 'Too many requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
});

// =============================================================================
// EXPORTS
// =============================================================================

module.exports = {
  // Error classes
  AppError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ConflictError,
  RateLimitError,
  
  // Validation functions
  validateEmail,
  validatePassword,
  validateRequired,
  validateCoordinates,
  
  // Middleware
  errorHandler,
  asyncHandler,
  notFoundHandler,
  validateSignup,
  validateLogin,
  validateProfileUpdate,
  validatePredictionFetch,
  authenticateToken,
  authorizeUser,
  
  // Rate limiting
  authLimiter,
  apiLimiter
}; 