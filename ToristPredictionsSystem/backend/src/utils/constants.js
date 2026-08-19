// Sri Lanka Tourism Constants

module.exports = {
  // Prediction types
  PREDICTION_TYPES: {
    TOURIST_ARRIVALS: 'tourist_arrivals',
    REVENUE: 'revenue',
    ROOMS: 'rooms',
    ALL: 'all'
  },

  // User roles
  USER_ROLES: {
    USER: 'user',
    ADMIN: 'admin'
  },

  // Sri Lankan months (Sinhala/Tamil names can be added)
  MONTHS: {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
  },

  // Month names
  MONTH_NAMES: [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ],

  // Sri Lankan tourism seasons
  SEASONS: {
    HIGH_SEASON: [12, 1, 2, 3], // December to March
    MID_SEASON: [7, 8], // July to August
    LOW_SEASON: [4, 5, 6, 9, 10, 11] // Rest of the year
  },

  // Currency
  DEFAULT_CURRENCY: 'USD',
  LKR_CURRENCY: 'LKR',

  // Default values
  DEFAULT_DOLLAR_RATE: 320,
  DEFAULT_PAGINATION_LIMIT: 50,

  // Tourist attractions in Sri Lanka
  ATTRACTIONS: [
    'Sigiriya Rock Fortress',
    'Temple of the Tooth',
    'Galle Fort',
    'Yala National Park',
    'Horton Plains',
    'Adam\'s Peak',
    'Arugam Bay',
    'Mirissa Beach'
  ],

  // Major tourist source countries
  TOP_SOURCE_COUNTRIES: [
    'India',
    'United Kingdom',
    'Germany',
    'France',
    'China',
    'Russia',
    'Australia',
    'United States',
    'Maldives',
    'Japan'
  ],

  // Response messages
  MESSAGES: {
    SUCCESS: {
      PREDICTION_CREATED: '🇱🇰 Prediction generated successfully for Sri Lankan tourism',
      LOGIN_SUCCESS: 'Login successful',
      REGISTER_SUCCESS: 'Registration successful',
      DATA_FETCHED: 'Data fetched successfully'
    },
    ERROR: {
      UNAUTHORIZED: 'Not authorized to access this route',
      INVALID_CREDENTIALS: 'Invalid credentials',
      SERVER_ERROR: 'Server error occurred',
      NOT_FOUND: 'Resource not found',
      VALIDATION_ERROR: 'Validation failed',
      ML_ERROR: 'Machine learning prediction failed'
    }
  },

  // HTTP status codes
  STATUS_CODES: {
    OK: 200,
    CREATED: 201,
    BAD_REQUEST: 400,
    UNAUTHORIZED: 401,
    FORBIDDEN: 403,
    NOT_FOUND: 404,
    SERVER_ERROR: 500
  },

  // Model accuracy thresholds
  ACCURACY_THRESHOLDS: {
    EXCELLENT: 0.90,
    GOOD: 0.80,
    FAIR: 0.70,
    POOR: 0.60
  },

  // Data ranges
  YEAR_RANGE: {
    MIN: 2000,
    MAX: 2050
  },

  // Feature count for ML model
  LAG_FEATURES: 12,

  // Sri Lanka specific
  COUNTRY_CODE: 'LK',
  COUNTRY_NAME: 'Sri Lanka',
  TOURISM_SLOGAN: 'Wonder of Asia',
  
  // Peak tourist months in Sri Lanka
  PEAK_MONTHS: [12, 1, 2], // December, January, February
  
  // Festival months affecting tourism
  FESTIVAL_MONTHS: {
    SINHALA_TAMIL_NEW_YEAR: 4, // April
    VESAK: 5, // May
    ESALA_PERAHERA: 7, // July/August
    DEEPAVALI: 10 // October/November
  }
};
