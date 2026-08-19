const { body, param, query } = require('express-validator');

// User validation
exports.registerValidation = [
  body('name')
    .trim()
    .notEmpty()
    .withMessage('Name is required')
    .isLength({ min: 2, max: 50 })
    .withMessage('Name must be between 2 and 50 characters'),
  
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email address'),
  
  body('password')
    .isLength({ min: 6 })
    .withMessage('Password must be at least 6 characters')
    .matches(/\d/)
    .withMessage('Password must contain at least one number'),
  
  body('organization')
    .optional()
    .trim()
];

exports.loginValidation = [
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email address'),
  
  body('password')
    .notEmpty()
    .withMessage('Password is required')
];

// Prediction validation
exports.predictionValidation = [
  body('year')
    .isInt({ min: 2000, max: 2050 })
    .withMessage('Year must be between 2000 and 2050'),
  
  body('month')
    .isInt({ min: 1, max: 12 })
    .withMessage('Month must be between 1 and 12'),
  
  body('dollarRate')
    .optional()
    .isFloat({ min: 0 })
    .withMessage('Dollar rate must be a positive number'),
  
  body('predictionType')
    .optional()
    .isIn(['tourist_arrivals', 'revenue', 'rooms', 'all'])
    .withMessage('Invalid prediction type')
];

exports.batchPredictionValidation = [
  body('startYear')
    .isInt({ min: 2000, max: 2050 })
    .withMessage('Start year must be between 2000 and 2050'),
  
  body('startMonth')
    .isInt({ min: 1, max: 12 })
    .withMessage('Start month must be between 1 and 12'),
  
  body('endYear')
    .isInt({ min: 2000, max: 2050 })
    .withMessage('End year must be between 2000 and 2050'),
  
  body('endMonth')
    .isInt({ min: 1, max: 12 })
    .withMessage('End month must be between 1 and 12'),
  
  body('dollarRate')
    .optional()
    .isFloat({ min: 0 })
    .withMessage('Dollar rate must be a positive number')
];

// Query validation
exports.yearQueryValidation = [
  query('year')
    .optional()
    .isInt({ min: 2000, max: 2050 })
    .withMessage('Year must be between 2000 and 2050')
];

exports.idParamValidation = [
  param('id')
    .isMongoId()
    .withMessage('Invalid ID format')
];

// Helper function to check if date is valid
exports.isValidDate = (year, month) => {
  const y = parseInt(year);
  const m = parseInt(month);
  
  if (isNaN(y) || isNaN(m)) return false;
  if (y < 2000 || y > 2050) return false;
  if (m < 1 || m > 12) return false;
  
  return true;
};

// Helper function to validate Sri Lankan months
exports.sriLankanMonths = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];

exports.getMonthName = (monthNumber) => {
  return exports.sriLankanMonths[monthNumber - 1] || 'Invalid Month';
};
