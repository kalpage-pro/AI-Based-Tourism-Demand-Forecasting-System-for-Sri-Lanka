const express = require('express');
const router = express.Router();

// @desc    Health check endpoint
// @route   GET /api/v1/health
// @access  Public
router.get('/', (req, res) => {
  res.status(200).json({
    success: true,
    message: '🇱🇰 Sri Lanka Tourist Prediction System is healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV
  });
});

module.exports = router;
