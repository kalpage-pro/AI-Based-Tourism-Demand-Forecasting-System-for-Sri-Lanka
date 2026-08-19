const mongoose = require('mongoose');

const historicalDataSchema = new mongoose.Schema({
  year: {
    type: Number,
    required: true
  },
  month: {
    type: Number,
    required: true,
    min: 1,
    max: 12
  },
  touristArrivals: {
    type: Number,
    required: true
  },
  revenue: Number,
  rooms: Number,
  dollarRate: Number,
  avgStayDuration: Number,
  topCountries: [String],
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Compound index for unique year-month combinations
historicalDataSchema.index({ year: 1, month: 1 }, { unique: true });

module.exports = mongoose.model('HistoricalData', historicalDataSchema);