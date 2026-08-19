const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth.middleware');
const {
  exportPredictionsCSV,
  exportPredictionsPDF,
  exportHistoricalData,
  getExportHistory
} = require('../controllers/export.controller');

// All routes require authentication
router.use(protect);

// Export predictions
router.get('/predictions/csv', exportPredictionsCSV);
router.get('/predictions/pdf', exportPredictionsPDF);

// Export historical data
router.get('/historical', exportHistoricalData);

// Export history
router.get('/history', getExportHistory);

module.exports = router;
