/**
 * WaveFarer Unified Configuration
 * ===============================
 * 
 * This file manages configuration for both Flask and Express services
 * to ensure consistency across the microservices architecture.
 */

const config = {
  // =============================================================================
  // ENVIRONMENT CONFIGURATION
  // =============================================================================
  
  environment: process.env.NODE_ENV || 'development',
  
  // =============================================================================
  // DATABASE CONFIGURATION
  // =============================================================================
  
  database: {
    // MongoDB configuration (shared between services)
    mongodb: {
      uri: process.env.MONGODB_URI || 'mongodb://localhost:27017/',
      options: {
        useNewUrlParser: true,
        useUnifiedTopology: true,
        serverSelectionTimeoutMS: 5000,
      }
    },
    
    // Database names for each service
    databases: {
      flask: 'wavefarer_predictions',  // Flask API database
      express: 'wavefarer_users'       // Express API database
    }
  },
  
  // =============================================================================
  // API CONFIGURATION
  // =============================================================================
  
  apis: {
    // Flask API (ML Services)
    flask: {
      port: process.env.FLASK_PORT || 5001,
      host: process.env.FLASK_HOST || '0.0.0.0',
      debug: process.env.FLASK_DEBUG === 'true',
      secret_key: process.env.FLASK_SECRET_KEY || 'dev-secret-key-change-in-production',
      cors_origins: [
        'http://localhost:3000',
        'http://localhost:5173',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:5173'
      ]
    },
    
    // Express API (User Services)
    express: {
      port: process.env.EXPRESS_PORT || 3000,
      host: process.env.EXPRESS_HOST || '0.0.0.0',
      jwt_secret: process.env.JWT_SECRET || 'your-jwt-secret-key',
      jwt_expires_in: '24h'
    }
  },
  
  // =============================================================================
  // SECURITY CONFIGURATION
  // =============================================================================
  
  security: {
    // Rate limiting
    rate_limit: {
      window_ms: 60 * 60 * 1000, // 1 hour
      max_requests: 100,          // 100 requests per hour
      message: 'Rate limit exceeded'
    },
    
    // CORS configuration
    cors: {
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowed_headers: ['Content-Type', 'Authorization']
    },
    
    // Security headers
    headers: {
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }
  },
  
  // =============================================================================
  // ML MODEL CONFIGURATION
  // =============================================================================
  
  ml: {
    // Model file paths
    models: {
      weather_forecast: 'models/weather_forecast_model_v2.pth',
      alert_model: 'alert_model.pkl'
    },
    
    // Model configuration
    config: {
      input_hours: 24,
      output_hours: 72,
      input_features: ['temperature', 'humidity', 'wind_direction', 'wind_speed'],
      target_features: ['temperature', 'wind_speed', 'wind_direction', 'wave_height', 'sea_surface_temp']
    }
  },
  
  // =============================================================================
  // LOGGING CONFIGURATION
  // =============================================================================
  
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    file: process.env.LOG_FILE || 'wavefarer.log',
    format: 'combined'
  },
  
  // =============================================================================
  // EXTERNAL API CONFIGURATION
  // =============================================================================
  
  external_apis: {
    // Weather APIs
    weather: {
      openweather: process.env.OPENWEATHER_API_KEY,
      marine: process.env.MARINE_API_KEY
    },
    
    // Geocoding API
    geocoding: {
      nominatim: 'https://nominatim.openstreetmap.org'
    }
  }
};

// =============================================================================
// VALIDATION FUNCTIONS
// =============================================================================

/**
 * Validate configuration
 */
function validateConfig() {
  const errors = [];
  
  // Check required environment variables
  if (!config.database.mongodb.uri) {
    errors.push('MONGODB_URI is required');
  }
  
  if (!config.apis.express.jwt_secret || config.apis.express.jwt_secret === 'your-jwt-secret-key') {
    errors.push('JWT_SECRET must be set in production');
  }
  
  if (config.environment === 'production' && config.apis.flask.secret_key === 'dev-secret-key-change-in-production') {
    errors.push('FLASK_SECRET_KEY must be changed in production');
  }
  
  return errors;
}

/**
 * Get service-specific configuration
 */
function getServiceConfig(service) {
  switch (service) {
    case 'flask':
      return {
        port: config.apis.flask.port,
        host: config.apis.flask.host,
        debug: config.apis.flask.debug,
        secret_key: config.apis.flask.secret_key,
        cors_origins: config.apis.flask.cors_origins,
        database: config.database,
        security: config.security,
        ml: config.ml,
        logging: config.logging
      };
      
    case 'express':
      return {
        port: config.apis.express.port,
        host: config.apis.express.host,
        jwt_secret: config.apis.express.jwt_secret,
        jwt_expires_in: config.apis.express.jwt_expires_in,
        database: config.database,
        security: config.security,
        logging: config.logging
      };
      
    default:
      throw new Error(`Unknown service: ${service}`);
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

module.exports = {
  config,
  validateConfig,
  getServiceConfig
}; 