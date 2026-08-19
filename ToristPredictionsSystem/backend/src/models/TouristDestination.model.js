const mongoose = require('mongoose');

const hotelSchema = new mongoose.Schema({
  name: { type: String, required: true },
  rating: { type: Number, min: 1, max: 5 },
  priceRange: { type: String, enum: ['budget', 'mid-range', 'luxury'] },
  pricePerNight: { type: Number },
  amenities: [String],
  contact: String,
  website: String,
  image: String
});

const flightSchema = new mongoose.Schema({
  airline: { type: String, required: true },
  from: { type: String, required: true },
  price: { type: Number },
  duration: String,
  frequency: String,
  isEconomical: { type: Boolean, default: false }
});

const touristDestinationSchema = new mongoose.Schema({
  name: {
    type: String,
    required: [true, 'Destination name is required'],
    trim: true,
    unique: true
  },
  region: {
    type: String,
    required: true,
    enum: ['North', 'South', 'East', 'West', 'Central', 'North Central', 'North Western', 'Sabaragamuwa', 'Uva']
  },
  description: {
    type: String,
    required: true
  },
  highlights: [String],
  bestTimeToVisit: {
    type: String
  },
  category: {
    type: String,
    enum: ['beach', 'cultural', 'wildlife', 'adventure', 'hill-country', 'historical', 'religious', 'nature'],
    required: true
  },
  popularity: {
    type: Number,
    default: 0,
    min: 0,
    max: 100
  },
  monthlyArrivals: {
    type: Map,
    of: Number,
    default: {}
  },
  yearlyArrivals: {
    type: Number,
    default: 0
  },
  averageStayDays: {
    type: Number,
    default: 2
  },
  hotels: [hotelSchema],
  flights: [flightSchema],
  images: [String],
  mainImage: String,
  coordinates: {
    latitude: Number,
    longitude: Number
  },
  ratings: {
    overall: { type: Number, default: 0, min: 0, max: 5 },
    totalReviews: { type: Number, default: 0 }
  },
  isActive: {
    type: Boolean,
    default: true
  },
  isFeatured: {
    type: Boolean,
    default: false
  },
  addedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
});

// Update updatedAt on save
touristDestinationSchema.pre('save', function(next) {
  this.updatedAt = Date.now();
  next();
});

// Index for search
touristDestinationSchema.index({ name: 'text', description: 'text', region: 'text' });
touristDestinationSchema.index({ category: 1 });
touristDestinationSchema.index({ popularity: -1 });
touristDestinationSchema.index({ isFeatured: 1 });

module.exports = mongoose.model('TouristDestination', touristDestinationSchema);
