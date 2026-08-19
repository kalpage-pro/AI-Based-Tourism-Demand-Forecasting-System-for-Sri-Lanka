const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth.middleware');
const {
  runScenario,
  getScenarioTemplates,
  whatIfAnalysis
} = require('../controllers/scenario.controller');

// All routes require authentication
router.use(protect);

// Run scenario simulation
router.post('/simulate', runScenario);

// Get predefined scenario templates
router.get('/templates', getScenarioTemplates);

// What-if analysis for single parameter
router.post('/what-if', whatIfAnalysis);

module.exports = router;
