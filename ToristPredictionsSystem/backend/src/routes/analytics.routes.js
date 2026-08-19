const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth.middleware');
const {
  getFeatureImportance,
  compareModels,
  getModelMetrics,
  getSeasonalPatterns,
  getPredictionAccuracy,
  getForecast,
  getYearlyTrends,
  getTourismDashboard,
  getArrivalsBreakdown
} = require('../controllers/analytics.controller');

// All routes require authentication
router.use(protect);

// Feature importance analysis
router.get('/feature-importance', getFeatureImportance);

// Model comparison
router.get('/models/compare', compareModels);

// Model evaluation metrics
router.get('/models/metrics', getModelMetrics);

// Seasonal patterns
router.get('/seasonal-patterns', getSeasonalPatterns);

// Prediction accuracy tracking
router.get('/prediction-accuracy', getPredictionAccuracy);

// 12-month forecast
router.get('/forecast', getForecast);

// Year-over-year trends
router.get('/yearly-trends', getYearlyTrends);

// Tourism dashboard with destination analytics
router.get('/tourism-dashboard', getTourismDashboard);

// Arrivals breakdown by category/region
router.get('/arrivals-breakdown', getArrivalsBreakdown);

module.exports = router;
