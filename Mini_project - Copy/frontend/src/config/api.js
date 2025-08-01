/**
 * WaveFarer API Configuration
 * ===========================
 * 
 * This file contains all API endpoint configurations for the frontend
 * to ensure consistent communication with both Flask and Express APIs.
 */

const API_CONFIG = {
  // ML Services (Flask API)
  ML_API: {
    baseURL: process.env.REACT_APP_ML_API_URL || 'http://localhost:5001',
    endpoints: {
      predict: '/api/predict',
      futurePredict: '/api/future-predict',
      health: '/api/health'
    }
  },
  
  // User Services (Express API)
  USER_API: {
    baseURL: process.env.REACT_APP_USER_API_URL || 'http://localhost:3000',
    endpoints: {
      signup: '/api/auth/signup',
      login: '/api/auth/login',
      getUser: (email) => `/api/users/${email}`,
      updateUser: (email) => `/api/users/${email}`,
      getPredictions: (email) => `/api/predictions/${email}`,
      fetchPrediction: '/api/predictions/fetch',
      health: '/api/health'
    }
  },
  
  // External APIs
  EXTERNAL: {
    weather: 'https://api.open-meteo.com/v1/forecast',
    marine: 'https://marine-api.open-meteo.com/v1/marine',
    geocoding: 'https://nominatim.openstreetmap.org/search'
  }
};

/**
 * API Service Class
 * =================
 * Provides methods for making API calls to both services
 */
class APIService {
  constructor() {
    this.mlBaseURL = API_CONFIG.ML_API.baseURL;
    this.userBaseURL = API_CONFIG.USER_API.baseURL;
  }

  /**
   * Make a request to the ML API (Flask)
   */
  async mlRequest(endpoint, options = {}) {
    const url = `${this.mlBaseURL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    };

    try {
      const response = await fetch(url, { ...defaultOptions, ...options });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(
          errorData.error?.message || `ML API Error: ${response.status} ${response.statusText}`,
          response.status,
          endpoint
        );
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new NetworkError();
      }
      
      throw new APIError(error.message, 500, endpoint);
    }
  }

  /**
   * Make a request to the User API (Express)
   */
  async userRequest(endpoint, options = {}) {
    const url = `${this.userBaseURL}${endpoint}`;
    const defaultOptions = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    };

    try {
      const response = await fetch(url, { ...defaultOptions, ...options });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new APIError(
          errorData.error?.message || `User API Error: ${response.status} ${response.statusText}`,
          response.status,
          endpoint
        );
      }
      
      return await response.json();
    } catch (error) {
      if (error instanceof APIError) {
        throw error;
      }
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new NetworkError();
      }
      
      throw new APIError(error.message, 500, endpoint);
    }
  }

  // =============================================================================
  // ML API METHODS
  // =============================================================================

  /**
   * Make a beach safety prediction
   */
  async makePrediction(weatherData) {
    return this.mlRequest(API_CONFIG.ML_API.endpoints.predict, {
      method: 'POST',
      body: JSON.stringify(weatherData)
    });
  }

  /**
   * Get 3-day weather forecast
   */
  async getForecast(lat, lon) {
    return this.mlRequest(API_CONFIG.ML_API.endpoints.futurePredict, {
      method: 'POST',
      body: JSON.stringify({ lat, lon })
    });
  }

  /**
   * Check ML API health
   */
  async checkMLHealth() {
    return this.mlRequest(API_CONFIG.ML_API.endpoints.health, {
      method: 'GET'
    });
  }

  // =============================================================================
  // USER API METHODS
  // =============================================================================

  /**
   * User registration
   */
  async signup(userData) {
    return this.userRequest(API_CONFIG.USER_API.endpoints.signup, {
      method: 'POST',
      body: JSON.stringify(userData)
    });
  }

  /**
   * User login
   */
  async login(credentials) {
    return this.userRequest(API_CONFIG.USER_API.endpoints.login, {
      method: 'POST',
      body: JSON.stringify(credentials)
    });
  }

  /**
   * Get user profile
   */
  async getUserProfile(email, token) {
    return this.userRequest(API_CONFIG.USER_API.endpoints.getUser(email), {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
  }

  /**
   * Update user profile
   */
  async updateUserProfile(email, profileData, token) {
    return this.userRequest(API_CONFIG.USER_API.endpoints.updateUser(email), {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify(profileData)
    });
  }

  /**
   * Get user's prediction history
   */
  async getUserPredictions(email, token) {
    return this.userRequest(API_CONFIG.USER_API.endpoints.getPredictions(email), {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
  }

  /**
   * Fetch specific prediction data
   */
  async fetchPrediction(userEmail, beachName) {
    const params = new URLSearchParams({
      userEmail,
      beachName
    });
    
    return this.userRequest(`${API_CONFIG.USER_API.endpoints.fetchPrediction}?${params}`, {
      method: 'GET'
    });
  }

  /**
   * Check User API health
   */
  async checkUserHealth() {
    return this.userRequest(API_CONFIG.USER_API.endpoints.health, {
      method: 'GET'
    });
  }

  // =============================================================================
  // EXTERNAL API METHODS
  // =============================================================================

  /**
   * Get weather data from external API
   */
  async getWeatherData(lat, lon) {
    const params = new URLSearchParams({
      latitude: lat,
      longitude: lon,
      current: 'temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,weather_code'
    });

    const response = await fetch(`${API_CONFIG.EXTERNAL.weather}?${params}`);
    if (!response.ok) {
      throw new Error(`Weather API Error: ${response.status}`);
    }
    return response.json();
  }

  /**
   * Get marine data from external API
   */
  async getMarineData(lat, lon) {
    const params = new URLSearchParams({
      latitude: lat,
      longitude: lon,
      current: 'wave_height,ocean_current_velocity,sea_surface_temperature'
    });

    const response = await fetch(`${API_CONFIG.EXTERNAL.marine}?${params}`);
    if (!response.ok) {
      throw new Error(`Marine API Error: ${response.status}`);
    }
    return response.json();
  }

  /**
   * Geocode location
   */
  async geocodeLocation(query) {
    const params = new URLSearchParams({
      format: 'json',
      q: query
    });

    const response = await fetch(`${API_CONFIG.EXTERNAL.geocoding}?${params}`);
    if (!response.ok) {
      throw new Error(`Geocoding API Error: ${response.status}`);
    }
    return response.json();
  }

  // =============================================================================
  // UTILITY METHODS
  // =============================================================================

  /**
   * Check all API health statuses
   */
  async checkAllHealth() {
    try {
      const [mlHealth, userHealth] = await Promise.all([
        this.checkMLHealth(),
        this.checkUserHealth()
      ]);

      return {
        ml: mlHealth,
        user: userHealth,
        allHealthy: mlHealth.status === 'healthy' && userHealth.status === 'healthy'
      };
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        ml: { status: 'error', message: 'ML API unavailable' },
        user: { status: 'error', message: 'User API unavailable' },
        allHealthy: false
      };
    }
  }

  /**
   * Get API configuration for debugging
   */
  getConfig() {
    return {
      ml: {
        baseURL: this.mlBaseURL,
        endpoints: API_CONFIG.ML_API.endpoints
      },
      user: {
        baseURL: this.userBaseURL,
        endpoints: API_CONFIG.USER_API.endpoints
      }
    };
  }
}

// Create and export a singleton instance
const apiService = new APIService();
export default apiService;
export { API_CONFIG }; 