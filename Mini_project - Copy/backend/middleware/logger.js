/**
 * WaveFarer Logging Middleware
 * ============================
 * 
 * Comprehensive logging system for tracking requests, errors,
 * and system events with different log levels and formats.
 */

const fs = require('fs');
const path = require('path');

// =============================================================================
// LOGGER CONFIGURATION
// =============================================================================

const LOG_LEVELS = {
  ERROR: 0,
  WARN: 1,
  INFO: 2,
  DEBUG: 3
};

const LOG_COLORS = {
  ERROR: '\x1b[31m', // Red
  WARN: '\x1b[33m',  // Yellow
  INFO: '\x1b[36m',  // Cyan
  DEBUG: '\x1b[35m', // Magenta
  RESET: '\x1b[0m'   // Reset
};

class Logger {
  constructor() {
    this.logLevel = process.env.LOG_LEVEL || 'INFO';
    this.logFile = process.env.LOG_FILE || 'wavefarer.log';
    this.logDir = path.join(process.cwd(), 'logs');
    
    // Create logs directory if it doesn't exist
    if (!fs.existsSync(this.logDir)) {
      fs.mkdirSync(this.logDir, { recursive: true });
    }
    
    this.logPath = path.join(this.logDir, this.logFile);
  }

  /**
   * Get current timestamp
   */
  getTimestamp() {
    return new Date().toISOString();
  }

  /**
   * Write log to file
   */
  writeToFile(level, message, data = {}) {
    const logEntry = {
      timestamp: this.getTimestamp(),
      level,
      message,
      data,
      service: 'wavefarer-user-api'
    };

    const logString = JSON.stringify(logEntry) + '\n';
    
    fs.appendFile(this.logPath, logString, (err) => {
      if (err) {
        console.error('Failed to write to log file:', err);
      }
    });
  }

  /**
   * Format console output
   */
  formatConsole(level, message, data = {}) {
    const color = LOG_COLORS[level] || LOG_COLORS.INFO;
    const reset = LOG_COLORS.RESET;
    const timestamp = this.getTimestamp();
    
    let output = `${color}[${timestamp}] ${level}:${reset} ${message}`;
    
    if (Object.keys(data).length > 0) {
      output += ` ${JSON.stringify(data)}`;
    }
    
    return output;
  }

  /**
   * Log message with specified level
   */
  log(level, message, data = {}) {
    if (LOG_LEVELS[level] <= LOG_LEVELS[this.logLevel]) {
      // Console output
      console.log(this.formatConsole(level, message, data));
      
      // File output
      this.writeToFile(level, message, data);
    }
  }

  /**
   * Error logging
   */
  error(message, data = {}) {
    this.log('ERROR', message, data);
  }

  /**
   * Warning logging
   */
  warn(message, data = {}) {
    this.log('WARN', message, data);
  }

  /**
   * Info logging
   */
  info(message, data = {}) {
    this.log('INFO', message, data);
  }

  /**
   * Debug logging
   */
  debug(message, data = {}) {
    this.log('DEBUG', message, data);
  }

  /**
   * Log HTTP request
   */
  logRequest(req, res, next) {
    const start = Date.now();
    
    // Log request start
    this.info('Request started', {
      method: req.method,
      url: req.url,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      contentLength: req.get('Content-Length')
    });

    // Override res.end to log response
    const originalEnd = res.end;
    res.end = function(chunk, encoding) {
      const duration = Date.now() - start;
      
      logger.info('Request completed', {
        method: req.method,
        url: req.url,
        statusCode: res.statusCode,
        duration: `${duration}ms`,
        contentLength: res.get('Content-Length')
      });
      
      originalEnd.call(this, chunk, encoding);
    };
    
    next();
  }

  /**
   * Log database operations
   */
  logDatabase(operation, collection, duration, success = true) {
    const level = success ? 'INFO' : 'ERROR';
    const message = success ? 'Database operation completed' : 'Database operation failed';
    
    this.log(level, message, {
      operation,
      collection,
      duration: `${duration}ms`,
      success
    });
  }

  /**
   * Log authentication events
   */
  logAuth(event, email, success = true) {
    const level = success ? 'INFO' : 'WARN';
    const message = success ? 'Authentication successful' : 'Authentication failed';
    
    this.log(level, message, {
      event,
      email,
      success
    });
  }

  /**
   * Log prediction events
   */
  logPrediction(operation, beachName, userEmail, success = true) {
    const level = success ? 'INFO' : 'ERROR';
    const message = success ? 'Prediction operation completed' : 'Prediction operation failed';
    
    this.log(level, message, {
      operation,
      beachName,
      userEmail,
      success
    });
  }

  /**
   * Log system events
   */
  logSystem(event, data = {}) {
    this.info(`System event: ${event}`, data);
  }

  /**
   * Log performance metrics
   */
  logPerformance(metric, value, unit = 'ms') {
    this.info('Performance metric', {
      metric,
      value: `${value}${unit}`
    });
  }

  /**
   * Log security events
   */
  logSecurity(event, data = {}) {
    this.warn(`Security event: ${event}`, data);
  }
}

// Create singleton logger instance
const logger = new Logger();

// =============================================================================
// REQUEST LOGGING MIDDLEWARE
// =============================================================================

/**
 * Request logging middleware
 */
function requestLogger(req, res, next) {
  logger.logRequest(req, res, next);
}

/**
 * Error logging middleware
 */
function errorLogger(err, req, res, next) {
  logger.error('Unhandled error', {
    message: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method,
    ip: req.ip
  });
  next(err);
}

// =============================================================================
// PERFORMANCE MONITORING
// =============================================================================

/**
 * Performance monitoring middleware
 */
function performanceMonitor(req, res, next) {
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
  
  next();
}

// =============================================================================
// SECURITY LOGGING
// =============================================================================

/**
 * Security event logging middleware
 */
function securityLogger(req, res, next) {
  // Log suspicious activities
  const suspiciousPatterns = [
    /\.\./, // Directory traversal
    /<script/i, // XSS attempts
    /union.*select/i, // SQL injection attempts
    /eval\(/i, // Code injection attempts
  ];
  
  const url = req.url;
  const userAgent = req.get('User-Agent') || '';
  
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(url) || pattern.test(userAgent)) {
      logger.logSecurity('suspicious_request', {
        url,
        userAgent,
        ip: req.ip,
        pattern: pattern.toString()
      });
      break;
    }
  }
  
  next();
}

// =============================================================================
// EXPORTS
// =============================================================================

module.exports = {
  logger,
  requestLogger,
  errorLogger,
  performanceMonitor,
  securityLogger
}; 