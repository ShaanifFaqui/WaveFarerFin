/**
 * WaveFarer Frontend Error Handling
 * ==================================
 * 
 * Comprehensive error handling system for the React frontend
 * with user-friendly messages, error logging, and recovery mechanisms.
 */

// =============================================================================
// ERROR TYPES
// =============================================================================

class FrontendError extends Error {
  constructor(message, type = 'GENERAL', code = null, recoverable = true) {
    super(message);
    this.name = 'FrontendError';
    this.type = type;
    this.code = code;
    this.recoverable = recoverable;
    this.timestamp = new Date().toISOString();
  }
}

class NetworkError extends FrontendError {
  constructor(message = 'Network connection failed', code = 'NETWORK_ERROR') {
    super(message, 'NETWORK', code, true);
    this.name = 'NetworkError';
  }
}

class APIError extends FrontendError {
  constructor(message, statusCode, endpoint) {
    super(message, 'API', statusCode, true);
    this.name = 'APIError';
    this.statusCode = statusCode;
    this.endpoint = endpoint;
  }
}

class ValidationError extends FrontendError {
  constructor(message, field = null) {
    super(message, 'VALIDATION', 'VALIDATION_ERROR', true);
    this.name = 'ValidationError';
    this.field = field;
  }
}

class AuthenticationError extends FrontendError {
  constructor(message = 'Authentication failed') {
    super(message, 'AUTH', 'AUTH_ERROR', true);
    this.name = 'AuthenticationError';
  }
}

class UserInputError extends FrontendError {
  constructor(message, field = null) {
    super(message, 'USER_INPUT', 'INPUT_ERROR', true);
    this.name = 'UserInputError';
    this.field = field;
  }
}

// =============================================================================
// ERROR MESSAGES
// =============================================================================

const ERROR_MESSAGES = {
  // Network errors
  NETWORK_ERROR: {
    title: 'Connection Error',
    message: 'Unable to connect to the server. Please check your internet connection and try again.',
    action: 'Retry'
  },
  
  // API errors
  API_ERROR: {
    title: 'Service Error',
    message: 'The service is temporarily unavailable. Please try again later.',
    action: 'Retry'
  },
  
  // Authentication errors
  AUTH_ERROR: {
    title: 'Authentication Error',
    message: 'Please log in again to continue.',
    action: 'Login'
  },
  
  // Validation errors
  VALIDATION_ERROR: {
    title: 'Invalid Input',
    message: 'Please check your input and try again.',
    action: 'Fix'
  },
  
  // User input errors
  INPUT_ERROR: {
    title: 'Invalid Input',
    message: 'Please provide valid information.',
    action: 'Correct'
  },
  
  // General errors
  GENERAL_ERROR: {
    title: 'Something went wrong',
    message: 'An unexpected error occurred. Please try again.',
    action: 'Retry'
  }
};

// =============================================================================
// ERROR HANDLER CLASS
// =============================================================================

class ErrorHandler {
  constructor() {
    this.errorLog = [];
    this.maxLogSize = 100;
    this.isDevelopment = process.env.NODE_ENV === 'development';
  }

  /**
   * Log error for debugging
   */
  logError(error, context = {}) {
    const errorEntry = {
      timestamp: new Date().toISOString(),
      error: {
        name: error.name,
        message: error.message,
        type: error.type,
        code: error.code,
        stack: this.isDevelopment ? error.stack : undefined
      },
      context,
      userAgent: navigator.userAgent,
      url: window.location.href
    };

    this.errorLog.push(errorEntry);
    
    // Keep log size manageable
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog.shift();
    }

    // Console log in development
    if (this.isDevelopment) {
      console.error('Error logged:', errorEntry);
    }

    // Send to external logging service in production
    if (!this.isDevelopment) {
      this.sendToLoggingService(errorEntry);
    }
  }

  /**
   * Send error to external logging service
   */
  sendToLoggingService(errorEntry) {
    // In a real application, you would send this to a logging service
    // like Sentry, LogRocket, or your own logging API
    fetch('/api/logs/error', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorEntry)
    }).catch(() => {
      // Silently fail if logging service is unavailable
    });
  }

  /**
   * Get user-friendly error message
   */
  getUserFriendlyMessage(error) {
    if (error instanceof NetworkError) {
      return ERROR_MESSAGES.NETWORK_ERROR;
    }
    
    if (error instanceof APIError) {
      switch (error.statusCode) {
        case 401:
          return {
            title: 'Authentication Required',
            message: 'Please log in to continue.',
            action: 'Login'
          };
        case 403:
          return {
            title: 'Access Denied',
            message: 'You don\'t have permission to perform this action.',
            action: 'Contact Support'
          };
        case 404:
          return {
            title: 'Not Found',
            message: 'The requested resource was not found.',
            action: 'Go Back'
          };
        case 429:
          return {
            title: 'Too Many Requests',
            message: 'Please wait a moment before trying again.',
            action: 'Wait'
          };
        case 500:
          return ERROR_MESSAGES.API_ERROR;
        default:
          return ERROR_MESSAGES.API_ERROR;
      }
    }
    
    if (error instanceof ValidationError) {
      return ERROR_MESSAGES.VALIDATION_ERROR;
    }
    
    if (error instanceof AuthenticationError) {
      return ERROR_MESSAGES.AUTH_ERROR;
    }
    
    if (error instanceof UserInputError) {
      return ERROR_MESSAGES.INPUT_ERROR;
    }
    
    return ERROR_MESSAGES.GENERAL_ERROR;
  }

  /**
   * Handle API errors
   */
  handleAPIError(response, endpoint) {
    let message = 'API request failed';
    
    try {
      const errorData = response.data;
      message = errorData.error?.message || errorData.message || message;
    } catch (e) {
      // If response parsing fails, use default message
    }
    
    const error = new APIError(message, response.status, endpoint);
    this.logError(error, { endpoint, statusCode: response.status });
    return error;
  }

  /**
   * Handle network errors
   */
  handleNetworkError(error, endpoint) {
    const networkError = new NetworkError();
    this.logError(networkError, { endpoint, originalError: error.message });
    return networkError;
  }

  /**
   * Handle validation errors
   */
  handleValidationError(message, field = null) {
    const validationError = new ValidationError(message, field);
    this.logError(validationError, { field });
    return validationError;
  }

  /**
   * Handle authentication errors
   */
  handleAuthError(message = 'Authentication failed') {
    const authError = new AuthenticationError(message);
    this.logError(authError);
    return authError;
  }

  /**
   * Handle user input errors
   */
  handleUserInputError(message, field = null) {
    const inputError = new UserInputError(message, field);
    this.logError(inputError, { field });
    return inputError;
  }

  /**
   * Validate user input
   */
  validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      throw this.handleValidationError('Please enter a valid email address', 'email');
    }
    return email.toLowerCase().trim();
  }

  validatePassword(password) {
    if (!password || password.length < 6) {
      throw this.handleValidationError('Password must be at least 6 characters long', 'password');
    }
    return password;
  }

  validateRequired(value, fieldName) {
    if (!value || (typeof value === 'string' && value.trim() === '')) {
      throw this.handleValidationError(`${fieldName} is required`, fieldName);
    }
    return value;
  }

  validateCoordinates(lat, lon) {
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);
    
    if (isNaN(latNum) || isNaN(lonNum)) {
      throw this.handleValidationError('Invalid coordinates format');
    }
    
    if (latNum < -90 || latNum > 90) {
      throw this.handleValidationError('Latitude must be between -90 and 90');
    }
    
    if (lonNum < -180 || lonNum > 180) {
      throw this.handleValidationError('Longitude must be between -180 and 180');
    }
    
    return { lat: latNum, lon: lonNum };
  }

  /**
   * Show error notification
   */
  showError(error, duration = 5000) {
    const message = this.getUserFriendlyMessage(error);
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = 'error-notification';
    notification.innerHTML = `
      <div class="error-notification-content">
        <h4>${message.title}</h4>
        <p>${message.message}</p>
        <button onclick="this.parentElement.parentElement.remove()">${message.action}</button>
      </div>
    `;
    
    // Add styles
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #f44336;
      color: white;
      padding: 16px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      z-index: 10000;
      max-width: 400px;
      animation: slideIn 0.3s ease-out;
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, duration);
  }

  /**
   * Show success notification
   */
  showSuccess(message, duration = 3000) {
    const notification = document.createElement('div');
    notification.className = 'success-notification';
    notification.innerHTML = `
      <div class="success-notification-content">
        <p>${message}</p>
      </div>
    `;
    
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #4caf50;
      color: white;
      padding: 16px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
      z-index: 10000;
      max-width: 400px;
      animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, duration);
  }

  /**
   * Get error log for debugging
   */
  getErrorLog() {
    return this.errorLog;
  }

  /**
   * Clear error log
   */
  clearErrorLog() {
    this.errorLog = [];
  }
}

// =============================================================================
// REACT ERROR BOUNDARY
// =============================================================================

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    const errorHandler = new ErrorHandler();
    errorHandler.logError(error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>We're sorry, but something unexpected happened. Please refresh the page to try again.</p>
          <button onClick={() => window.location.reload()}>
            Refresh Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  FrontendError,
  NetworkError,
  APIError,
  ValidationError,
  AuthenticationError,
  UserInputError,
  ErrorHandler,
  ErrorBoundary
};

// Create singleton error handler instance
export const errorHandler = new ErrorHandler(); 