const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  predictionType: {
    type: String,
    enum: ['tourist_arrivals', 'revenue', 'rooms', 'all'],
    required: true
  },
  inputData: {
    year: {
      type: Number,
      required: true,
      min: 2000,
      max: 2050
    },
    month: {
      type: Number,
      required: true,
      min: 1,
      max: 12
    },
    dollarRate: Number,
    additionalFeatures: mongoose.Schema.Types.Mixed
  },
  predictions: {
    touristArrivals: {
      value: Number,
      confidence: Number
    },
    revenue: {
      value: Number,
      currency: {
        type: String,
        default: 'USD'
      },
      confidence: Number
    },
    rooms: {
      value: Number,
      confidence: Number
    }
  },
  metadata: {
    modelVersion: String,
    processingTime: Number,
    accuracy: Number
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

// Index for faster queries
predictionSchema.index({ user: 1, createdAt: -1 });
predictionSchema.index({ 'inputData.year': 1, 'inputData.month': 1 });

module.exports = mongoose.model('Prediction', predictionSchema);