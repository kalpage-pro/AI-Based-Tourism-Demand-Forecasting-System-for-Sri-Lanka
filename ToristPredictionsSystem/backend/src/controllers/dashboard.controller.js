const Prediction = require('../models/Prediction.model');
const HistoricalData = require('../models/HistoricalData.model');

// @desc    Get dashboard statistics
// @route   GET /api/v1/dashboard/stats
// @access  Private
exports.getDashboardStats = async (req, res) => {
  try {
    const userId = req.user.id;
    const currentYear = new Date().getFullYear();

    // Get total predictions by user
    const totalPredictions = await Prediction.countDocuments({ user: userId });

    // Get recent predictions (last 5)
    const recentPredictions = await Prediction.find({ user: userId })
      .sort({ createdAt: -1 })
      .limit(5)
      .select('predictionType inputData predictions createdAt');

    // Get average tourist arrivals prediction for current year
    const yearPredictions = await Prediction.find({
      user: userId,
      'inputData.year': currentYear
    });

    const avgTouristArrivals = yearPredictions.length > 0
      ? yearPredictions.reduce((sum, p) => sum + (p.predictions.touristArrivals?.value || 0), 0) / yearPredictions.length
      : 0;

    // Get historical data summary
    const historicalCount = await HistoricalData.countDocuments();
    const latestHistorical = await HistoricalData.findOne()
      .sort({ year: -1, month: -1 })
      .select('year month touristArrivals revenue');

    res.status(200).json({
      success: true,
      data: {
        totalPredictions,
        recentPredictions,
        currentYearStats: {
          year: currentYear,
          avgTouristArrivals: Math.round(avgTouristArrivals),
          totalPredictions: yearPredictions.length
        },
        historicalData: {
          totalRecords: historicalCount,
          latest: latestHistorical
        }
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get monthly trends
// @route   GET /api/v1/dashboard/trends
// @access  Private
exports.getMonthlyTrends = async (req, res) => {
  try {
    const { year } = req.query;
    const targetYear = year ? parseInt(year) : new Date().getFullYear();

    // Get historical data for the year
    const historicalData = await HistoricalData.find({
      year: targetYear
    }).sort({ month: 1 });

    // Get predictions for the year
    const predictions = await Prediction.find({
      user: req.user.id,
      'inputData.year': targetYear
    }).sort({ 'inputData.month': 1 });

    // Format data for charts
    const monthlyData = [];
    for (let month = 1; month <= 12; month++) {
      const historical = historicalData.find(d => d.month === month);
      const prediction = predictions.find(p => p.inputData.month === month);

      monthlyData.push({
        month,
        monthName: new Date(2000, month - 1).toLocaleString('default', { month: 'long' }),
        historical: historical ? {
          touristArrivals: historical.touristArrivals,
          revenue: historical.revenue,
          rooms: historical.rooms
        } : null,
        predicted: prediction ? {
          touristArrivals: prediction.predictions.touristArrivals?.value,
          revenue: prediction.predictions.revenue?.value,
          rooms: prediction.predictions.rooms?.value
        } : null
      });
    }

    res.status(200).json({
      success: true,
      data: {
        year: targetYear,
        monthlyData
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get comparison data
// @route   GET /api/v1/dashboard/compare
// @access  Private
exports.getComparisonData = async (req, res) => {
  try {
    const currentYear = new Date().getFullYear();
    const lastYear = currentYear - 1;

    // Get data for both years
    const currentYearData = await HistoricalData.find({ year: currentYear });
    const lastYearData = await HistoricalData.find({ year: lastYear });

    const comparison = {
      currentYear: {
        year: currentYear,
        totalArrivals: currentYearData.reduce((sum, d) => sum + d.touristArrivals, 0),
        avgMonthlyArrivals: currentYearData.length > 0 
          ? currentYearData.reduce((sum, d) => sum + d.touristArrivals, 0) / currentYearData.length 
          : 0
      },
      lastYear: {
        year: lastYear,
        totalArrivals: lastYearData.reduce((sum, d) => sum + d.touristArrivals, 0),
        avgMonthlyArrivals: lastYearData.length > 0 
          ? lastYearData.reduce((sum, d) => sum + d.touristArrivals, 0) / lastYearData.length 
          : 0
      }
    };

    // Calculate growth rate
    if (comparison.lastYear.totalArrivals > 0) {
      comparison.growthRate = (
        ((comparison.currentYear.totalArrivals - comparison.lastYear.totalArrivals) / 
        comparison.lastYear.totalArrivals) * 100
      ).toFixed(2);
    } else {
      comparison.growthRate = 0;
    }

    res.status(200).json({
      success: true,
      data: comparison
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};