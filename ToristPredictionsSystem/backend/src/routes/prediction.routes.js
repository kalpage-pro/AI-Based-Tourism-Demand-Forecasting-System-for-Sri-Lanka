const express = require('express');
const { body } = require('express-validator');
const {
  createPrediction,
  getPredictions,
  getPredictionById,
  batchPredictions
} = require('../controllers/prediction.controller');
const { protect } = require('../middleware/auth.middleware');
const { validate } = require('../middleware/validation.middleware');

const router = express.Router();

// Validation rules
const predictionValidation = [
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
  validate
];

const batchValidation = [
  body('startYear').isInt({ min: 2000 }).withMessage('Start year is required'),
  body('startMonth').isInt({ min: 1, max: 12 }).withMessage('Start month must be between 1 and 12'),
  body('endYear').isInt({ min: 2000 }).withMessage('End year is required'),
  body('endMonth').isInt({ min: 1, max: 12 }).withMessage('End month must be between 1 and 12'),
  validate
];

// All routes require authentication
router.use(protect);

router.post('/', predictionValidation, createPrediction);
router.get('/', getPredictions);
router.get('/:id', getPredictionById);
router.post('/batch', batchValidation, batchPredictions);

module.exports = router;