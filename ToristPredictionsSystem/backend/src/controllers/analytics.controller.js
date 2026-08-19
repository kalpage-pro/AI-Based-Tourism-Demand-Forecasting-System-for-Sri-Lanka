const { spawn } = require('child_process');
const path = require('path');
const Prediction = require('../models/Prediction.model');
const HistoricalData = require('../models/HistoricalData.model');
const TouristDestination = require('../models/TouristDestination.model');
const mlConfig = require('../config/ml.config');

// Get feature importance for all models
exports.getFeatureImportance = async (req, res, next) => {
  try {
    const { modelType = 'all' } = req.query;
    
    const result = await runPythonScript('feature_importance.py', ['--type', modelType]);
    
    res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    next(error);
  }
};

// Compare model performances
exports.compareModels = async (req, res, next) => {
  try {
    const result = await runPythonScript('evaluate_models.py', ['--action', 'compare']);
    
    res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    next(error);
  }
};

// Get model evaluation metrics
exports.getModelMetrics = async (req, res, next) => {
  try {
    const { model = 'all' } = req.query;
    
    const result = await runPythonScript('evaluate_models.py', ['--action', 'metrics', '--model', model]);
    
    res.status(200).json({
      success: true,
      data: result
    });
  } catch (error) {
    next(error);
  }
};

// Get seasonal patterns analysis
exports.getSeasonalPatterns = async (req, res, next) => {
  try {
    // Get historical data aggregated by month
    const seasonalData = await HistoricalData.aggregate([
      {
        $group: {
          _id: '$month',
          avgArrivals: { $avg: '$touristArrivals' },
          avgRevenue: { $avg: '$revenue' },
          avgOccupancy: { $avg: '$rooms' },
          count: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Calculate seasonal indices
    const totalAvgArrivals = seasonalData.reduce((sum, m) => sum + m.avgArrivals, 0) / 12;
    
    const patterns = seasonalData.map(month => ({
      month: month._id,
      monthName: getMonthName(month._id),
      avgArrivals: Math.round(month.avgArrivals),
      avgRevenue: Math.round(month.avgRevenue),
      avgOccupancy: month.avgOccupancy ? Math.round(month.avgOccupancy * 100) / 100 : null,
      seasonalIndex: totalAvgArrivals > 0 ? Math.round((month.avgArrivals / totalAvgArrivals) * 100) / 100 : 1,
      season: getSeason(month._id)
    }));

    // Identify peak and low seasons
    const sorted = [...patterns].sort((a, b) => b.avgArrivals - a.avgArrivals);
    const peakMonths = sorted.slice(0, 3).map(m => m.monthName);
    const lowMonths = sorted.slice(-3).map(m => m.monthName);

    res.status(200).json({
      success: true,
      data: {
        monthly: patterns,
        peakSeason: peakMonths,
        lowSeason: lowMonths,
        insights: generateSeasonalInsights(patterns)
      }
    });
  } catch (error) {
    next(error);
  }
};

// Get prediction accuracy over time
exports.getPredictionAccuracy = async (req, res, next) => {
  try {
    // Get predictions that have corresponding historical data
    const predictions = await Prediction.find({
      'predictions.touristArrivals.value': { $exists: true }
    }).sort({ createdAt: -1 }).limit(100);

    const accuracyData = [];

    for (const pred of predictions) {
      const historical = await HistoricalData.findOne({
        year: pred.inputData.year,
        month: pred.inputData.month
      });

      if (historical && historical.touristArrivals) {
        const predicted = pred.predictions.touristArrivals?.value;
        if (predicted) {
          const actual = historical.touristArrivals;
          const error = Math.abs(predicted - actual);
          const mape = (error / actual) * 100;
          
          accuracyData.push({
            year: pred.inputData.year,
            month: pred.inputData.month,
            predicted: Math.round(predicted),
            actual: actual,
            error: Math.round(error),
            mape: Math.round(mape * 100) / 100,
            accuracy: Math.round((100 - mape) * 100) / 100
          });
        }
      }
    }

    // Calculate overall metrics
    const avgMape = accuracyData.length > 0 
      ? accuracyData.reduce((sum, d) => sum + d.mape, 0) / accuracyData.length 
      : 0;

    res.status(200).json({
      success: true,
      data: {
        comparisons: accuracyData.slice(0, 20),
        overallAccuracy: Math.round((100 - avgMape) * 100) / 100,
        totalComparisons: accuracyData.length,
        avgMape: Math.round(avgMape * 100) / 100
      }
    });
  } catch (error) {
    next(error);
  }
};

// Get forecast for next 12 months
exports.getForecast = async (req, res, next) => {
  try {
    const { startYear, startMonth, model = 'rf' } = req.query;
    
    const year = parseInt(startYear) || new Date().getFullYear();
    const month = parseInt(startMonth) || new Date().getMonth() + 1;
    
    const forecasts = [];
    let currentYear = year;
    let currentMonth = month;
    
    for (let i = 0; i < 12; i++) {
      const result = await runPythonScript('predict.py', [
        '--year', currentYear.toString(),
        '--month', currentMonth.toString(),
        '--model', model,
        '--type', 'all'
      ]);
      
      forecasts.push({
        year: currentYear,
        month: currentMonth,
        monthName: getMonthName(currentMonth),
        predictions: result
      });
      
      currentMonth++;
      if (currentMonth > 12) {
        currentMonth = 1;
        currentYear++;
      }
    }
    
    res.status(200).json({
      success: true,
      data: forecasts
    });
  } catch (error) {
    next(error);
  }
};

// Get year-over-year trends
exports.getYearlyTrends = async (req, res, next) => {
  try {
    const yearlyData = await HistoricalData.aggregate([
      {
        $group: {
          _id: '$year',
          totalArrivals: { $sum: '$touristArrivals' },
          totalRevenue: { $sum: '$revenue' },
          avgOccupancy: { $avg: '$rooms' },
          months: { $sum: 1 }
        }
      },
      { $sort: { _id: 1 } }
    ]);

    // Calculate year-over-year growth
    const trends = yearlyData.map((year, index) => {
      const prevYear = index > 0 ? yearlyData[index - 1] : null;
      return {
        year: year._id,
        totalArrivals: year.totalArrivals,
        totalRevenue: year.totalRevenue,
        avgOccupancy: year.avgOccupancy ? Math.round(year.avgOccupancy * 100) / 100 : null,
        arrivalsGrowth: prevYear 
          ? Math.round(((year.totalArrivals - prevYear.totalArrivals) / prevYear.totalArrivals) * 10000) / 100 
          : null,
        revenueGrowth: prevYear && prevYear.totalRevenue
          ? Math.round(((year.totalRevenue - prevYear.totalRevenue) / prevYear.totalRevenue) * 10000) / 100
          : null
      };
    });

    res.status(200).json({
      success: true,
      data: trends
    });
  } catch (error) {
    next(error);
  }
};

// Helper function to run Python scripts
function runPythonScript(scriptName, args) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(mlConfig.scriptsPath, scriptName);
    
    const pythonProcess = spawn(mlConfig.pythonPath, [scriptPath, ...args], {
      cwd: mlConfig.scriptsPath,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let dataString = '';
    let errorString = '';

    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(dataString);
          resolve(result);
        } catch (e) {
          resolve({ raw: dataString });
        }
      } else {
        reject(new Error(`Python script failed: ${errorString}`));
      }
    });

    pythonProcess.on('error', (err) => {
      reject(new Error(`Failed to run Python: ${err.message}`));
    });
  });
}

function getMonthName(month) {
  const months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December'];
  return months[month] || '';
}

function getSeason(month) {
  if (month >= 12 || month <= 2) return 'Peak (Winter)';
  if (month >= 3 && month <= 5) return 'Shoulder (Spring)';
  if (month >= 6 && month <= 8) return 'Low (Monsoon)';
  return 'Shoulder (Autumn)';
}

function generateSeasonalInsights(patterns) {
  const insights = [];
  
  const maxMonth = patterns.reduce((max, p) => p.avgArrivals > max.avgArrivals ? p : max);
  const minMonth = patterns.reduce((min, p) => p.avgArrivals < min.avgArrivals ? p : min);
  
  insights.push(`Peak tourism month is ${maxMonth.monthName} with average ${maxMonth.avgArrivals.toLocaleString()} arrivals`);
  insights.push(`Lowest tourism month is ${minMonth.monthName} with average ${minMonth.avgArrivals.toLocaleString()} arrivals`);
  
  const variance = ((maxMonth.avgArrivals - minMonth.avgArrivals) / minMonth.avgArrivals * 100).toFixed(1);
  insights.push(`Seasonal variation is ${variance}% between peak and low seasons`);
  
  return insights;
}

// Get comprehensive tourism dashboard data
exports.getTourismDashboard = async (req, res, next) => {
  try {
    // Get destination analytics
    const [
      categoryStats,
      regionStats,
      topDestinations,
      featuredDestinations,
      overallStats
    ] = await Promise.all([
      // Arrivals by category for pie chart
      TouristDestination.aggregate([
        { $match: { isActive: true } },
        {
          $group: {
            _id: '$category',
            totalArrivals: { $sum: '$yearlyArrivals' },
            avgPopularity: { $avg: '$popularity' },
            count: { $sum: 1 }
          }
        },
        { $sort: { totalArrivals: -1 } }
      ]),
      // Arrivals by region for pie chart
      TouristDestination.aggregate([
        { $match: { isActive: true } },
        {
          $group: {
            _id: '$region',
            totalArrivals: { $sum: '$yearlyArrivals' },
            avgPopularity: { $avg: '$popularity' },
            count: { $sum: 1 }
          }
        },
        { $sort: { totalArrivals: -1 } }
      ]),
      // Top 10 destinations
      TouristDestination.find({ isActive: true })
        .sort({ yearlyArrivals: -1 })
        .limit(10)
        .select('name region category yearlyArrivals popularity ratings mainImage'),
      // Featured destinations
      TouristDestination.find({ isActive: true, isFeatured: true })
        .sort({ popularity: -1 })
        .limit(6)
        .select('name region category description yearlyArrivals popularity ratings mainImage highlights'),
      // Overall stats
      TouristDestination.aggregate([
        { $match: { isActive: true } },
        {
          $group: {
            _id: null,
            totalDestinations: { $sum: 1 },
            totalYearlyArrivals: { $sum: '$yearlyArrivals' },
            avgPopularity: { $avg: '$popularity' },
            avgRating: { $avg: '$ratings.overall' },
            avgStayDays: { $avg: '$averageStayDays' }
          }
        }
      ])
    ]);

    // Format category data for pie chart
    const categoryPieData = categoryStats.map(cat => ({
      name: cat._id ? cat._id.charAt(0).toUpperCase() + cat._id.slice(1).replace('-', ' ') : 'Other',
      value: cat.totalArrivals,
      count: cat.count,
      avgPopularity: Math.round(cat.avgPopularity)
    }));

    // Format region data for pie chart
    const regionPieData = regionStats.map(reg => ({
      name: reg._id || 'Unknown',
      value: reg.totalArrivals,
      count: reg.count,
      avgPopularity: Math.round(reg.avgPopularity)
    }));

    // Get historical yearly comparison
    const yearlyComparison = await HistoricalData.aggregate([
      {
        $group: {
          _id: '$year',
          totalArrivals: { $sum: '$touristArrivals' },
          avgRevenue: { $avg: '$revenue' }
        }
      },
      { $sort: { _id: 1 } },
      { $limit: 10 }
    ]);

    res.status(200).json({
      success: true,
      data: {
        overview: overallStats[0] || {
          totalDestinations: 0,
          totalYearlyArrivals: 0,
          avgPopularity: 0,
          avgRating: 0,
          avgStayDays: 0
        },
        charts: {
          categoryPieData,
          regionPieData,
          yearlyComparison
        },
        topDestinations,
        featuredDestinations
      }
    });
  } catch (error) {
    next(error);
  }
};

// Get arrivals breakdown for charts
exports.getArrivalsBreakdown = async (req, res, next) => {
  try {
    const { type = 'category' } = req.query;

    let groupField = '$category';
    if (type === 'region') groupField = '$region';

    const breakdown = await TouristDestination.aggregate([
      { $match: { isActive: true } },
      {
        $group: {
          _id: groupField,
          totalArrivals: { $sum: '$yearlyArrivals' },
          destinations: { $push: { name: '$name', arrivals: '$yearlyArrivals' } },
          count: { $sum: 1 }
        }
      },
      { $sort: { totalArrivals: -1 } }
    ]);

    // Calculate percentages
    const total = breakdown.reduce((sum, item) => sum + item.totalArrivals, 0);
    const formattedData = breakdown.map(item => ({
      name: item._id ? item._id.charAt(0).toUpperCase() + item._id.slice(1).replace('-', ' ') : 'Other',
      value: item.totalArrivals,
      percentage: total > 0 ? Math.round((item.totalArrivals / total) * 1000) / 10 : 0,
      count: item.count,
      topDestinations: item.destinations.sort((a, b) => b.arrivals - a.arrivals).slice(0, 3)
    }));

    res.status(200).json({
      success: true,
      data: {
        type,
        total,
        breakdown: formattedData
      }
    });
  } catch (error) {
    next(error);
  }
};
