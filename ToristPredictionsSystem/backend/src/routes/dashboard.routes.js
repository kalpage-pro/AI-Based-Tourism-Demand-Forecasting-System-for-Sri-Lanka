const express = require('express');
const {
  getDashboardStats,
  getMonthlyTrends,
  getComparisonData
} = require('../controllers/dashboard.controller');
const { protect } = require('../middleware/auth.middleware');

const router = express.Router();

// All routes require authentication
router.use(protect);

router.get('/stats', getDashboardStats);
router.get('/trends', getMonthlyTrends);
router.get('/compare', getComparisonData);

module.exports = router;
