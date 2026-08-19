const Prediction = require('../models/Prediction.model');
const HistoricalData = require('../models/HistoricalData.model');

class PredictionService {
  // Get prediction statistics
  async getPredictionStatistics(userId, filters = {}) {
    const query = { user: userId };

    if (filters.year) {
      query['inputData.year'] = filters.year;
    }

    if (filters.predictionType) {
      query.predictionType = filters.predictionType;
    }

    const predictions = await Prediction.find(query);

    return {
      total: predictions.length,
      byType: this.groupByType(predictions),
      averages: this.calculateAverages(predictions),
      monthlyDistribution: this.getMonthlyDistribution(predictions)
    };
  }

  // Group predictions by type
  groupByType(predictions) {
    const grouped = {
      tourist_arrivals: 0,
      revenue: 0,
      rooms: 0,
      all: 0
    };

    predictions.forEach(pred => {
      grouped[pred.predictionType]++;
    });

    return grouped;
  }

  // Calculate averages
  calculateAverages(predictions) {
    if (predictions.length === 0) {
      return { touristArrivals: 0, revenue: 0, rooms: 0 };
    }

    const sums = predictions.reduce((acc, pred) => {
      acc.touristArrivals += pred.predictions.touristArrivals?.value || 0;
      acc.revenue += pred.predictions.revenue?.value || 0;
      acc.rooms += pred.predictions.rooms?.value || 0;
      return acc;
    }, { touristArrivals: 0, revenue: 0, rooms: 0 });

    return {
      touristArrivals: Math.round(sums.touristArrivals / predictions.length),
      revenue: Math.round(sums.revenue / predictions.length),
      rooms: Math.round(sums.rooms / predictions.length)
    };
  }

  // Get monthly distribution
  getMonthlyDistribution(predictions) {
    const distribution = Array(12).fill(0);

    predictions.forEach(pred => {
      const month = pred.inputData.month - 1;
      distribution[month]++;
    });

    return distribution;
  }

  // Compare prediction with historical data
  async comparePredictionWithHistorical(year, month) {
    const historical = await HistoricalData.findOne({ year, month });
    const predictions = await Prediction.find({
      'inputData.year': year,
      'inputData.month': month
    });

    if (!historical || predictions.length === 0) {
      return null;
    }

    const avgPrediction = this.calculateAverages(predictions);

    return {
      historical: {
        touristArrivals: historical.touristArrivals,
        revenue: historical.revenue,
        rooms: historical.rooms
      },
      predicted: avgPrediction,
      accuracy: {
        touristArrivals: this.calculateAccuracy(
          historical.touristArrivals,
          avgPrediction.touristArrivals
        ),
        revenue: this.calculateAccuracy(
          historical.revenue,
          avgPrediction.revenue
        ),
        rooms: this.calculateAccuracy(
          historical.rooms,
          avgPrediction.rooms
        )
      }
    };
  }

  // Calculate accuracy percentage
  calculateAccuracy(actual, predicted) {
    if (!actual || !predicted) return 0;
    const error = Math.abs(actual - predicted);
    const accuracy = ((1 - error / actual) * 100);
    return Math.max(0, Math.min(100, accuracy)).toFixed(2);
  }

  // Get forecast for next N months
  async getForecast(startYear, startMonth, numberOfMonths, userId) {
    const forecasts = [];
    let currentYear = startYear;
    let currentMonth = startMonth;

    for (let i = 0; i < numberOfMonths; i++) {
      const predictions = await Prediction.find({
        user: userId,
        'inputData.year': currentYear,
        'inputData.month': currentMonth
      }).sort({ createdAt: -1 }).limit(1);

      if (predictions.length > 0) {
        forecasts.push({
          year: currentYear,
          month: currentMonth,
          predictions: predictions[0].predictions
        });
      }

      // Move to next month
      currentMonth++;
      if (currentMonth > 12) {
        currentMonth = 1;
        currentYear++;
      }
    }

    return forecasts;
  }
}

module.exports = new PredictionService();