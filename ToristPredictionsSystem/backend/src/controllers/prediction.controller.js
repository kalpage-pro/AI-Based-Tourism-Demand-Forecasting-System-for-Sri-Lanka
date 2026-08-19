const Prediction = require('../models/Prediction.model');
const mlService = require('../services/ml.service');
const { predictionTypes } = require('../config/ml.config');

// @desc    Make a new prediction
// @route   POST /api/v1/predictions
// @access  Private
exports.createPrediction = async (req, res) => {
  try {
    const { 
      year, 
      month, 
      dollarRate, 
      predictionType,
      modelChoice,
      // Advanced features
      apparentTemperature,
      sunshine,
      rain,
      precipitationHours,
      numEstablishments,
      numRooms,
      airfareIndex,
      cpi,
      arrivalsLag1,
      arrivalsLag2,
      arrivalsLag3,
      arrivalsLag12,
      arrivalsRoll3,
      arrivalsRoll6,
      arrivalsStd3,
      arrivalsYoy,
      revenueLag1,
      hotelOccupancyRate
    } = req.body;

    // Validate input
    if (!year || !month) {
      return res.status(400).json({
        success: false,
        message: 'Please provide year and month'
      });
    }

    // Call ML service to get predictions
    const mlResults = await mlService.predict({
      year,
      month,
      dollarRate: dollarRate || 320,
      predictionType: predictionType || predictionTypes.ALL,
      modelChoice: modelChoice || 'rf',
      apparentTemperature,
      sunshine,
      rain,
      precipitationHours,
      numEstablishments,
      numRooms,
      airfareIndex,
      cpi,
      arrivalsLag1,
      arrivalsLag2,
      arrivalsLag3,
      arrivalsLag12,
      arrivalsRoll3,
      arrivalsRoll6,
      arrivalsStd3,
      arrivalsYoy,
      revenueLag1,
      hotelOccupancyRate
    });

    // Save prediction to database
    const prediction = await Prediction.create({
      user: req.user.id,
      predictionType: predictionType || predictionTypes.ALL,
      inputData: {
        year,
        month,
        dollarRate: dollarRate || 320,
        modelChoice: modelChoice || 'rf'
      },
      predictions: mlResults.predictions,
      metadata: mlResults.metadata
    });

    res.status(201).json({
      success: true,
      data: prediction,
      message: '🇱🇰 Prediction generated successfully for Sri Lankan tourism'
    });
  } catch (error) {
    console.error('Prediction Error:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to generate prediction',
      error: error.message
    });
  }
};

// @desc    Get user's prediction history
// @route   GET /api/v1/predictions
// @access  Private
exports.getPredictions = async (req, res) => {
  try {
    const predictions = await Prediction.find({ user: req.user.id })
      .sort({ createdAt: -1 })
      .limit(50);

    res.status(200).json({
      success: true,
      count: predictions.length,
      data: predictions
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get prediction by ID
// @route   GET /api/v1/predictions/:id
// @access  Private
exports.getPredictionById = async (req, res) => {
  try {
    const prediction = await Prediction.findById(req.params.id);

    if (!prediction) {
      return res.status(404).json({
        success: false,
        message: 'Prediction not found'
      });
    }

    // Check if user owns the prediction
    if (prediction.user.toString() !== req.user.id && req.user.role !== 'admin') {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to access this prediction'
      });
    }

    res.status(200).json({
      success: true,
      data: prediction
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Batch predictions for multiple months
// @route   POST /api/v1/predictions/batch
// @access  Private
exports.batchPredictions = async (req, res) => {
  try {
    const { startYear, startMonth, endYear, endMonth, dollarRate } = req.body;

    const results = await mlService.batchPredict({
      startYear,
      startMonth,
      endYear,
      endMonth,
      dollarRate: dollarRate || 200
    });

    // Save all predictions
    const predictions = await Prediction.insertMany(
      results.map(result => ({
        user: req.user.id,
        predictionType: predictionTypes.ALL,
        inputData: result.inputData,
        predictions: result.predictions,
        metadata: result.metadata
      }))
    );

    res.status(201).json({
      success: true,
      count: predictions.length,
      data: predictions
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};