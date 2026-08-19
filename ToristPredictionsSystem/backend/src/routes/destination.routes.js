const express = require('express');
const router = express.Router();
const { protect, authorize } = require('../middleware/auth.middleware');
const {
  getAllDestinations,
  getDestination,
  createDestination,
  updateDestination,
  deleteDestination,
  getFeaturedDestinations,
  getDestinationAnalytics,
  addHotel,
  updateHotel,
  deleteHotel,
  addFlight,
  deleteFlight,
  toggleFeatured,
  getEconomicalFlights,
  getBestHotels
} = require('../controllers/destination.controller');

// All routes require authentication
router.use(protect);

// Public routes for users (read-only)
router.get('/', getAllDestinations);
router.get('/featured', getFeaturedDestinations);
router.get('/analytics', getDestinationAnalytics);
router.get('/economical-flights', getEconomicalFlights);
router.get('/best-hotels', getBestHotels);
router.get('/:id', getDestination);

// Admin-only routes
router.post('/', authorize('admin'), createDestination);
router.put('/:id', authorize('admin'), updateDestination);
router.delete('/:id', authorize('admin'), deleteDestination);
router.patch('/:id/toggle-featured', authorize('admin'), toggleFeatured);

// Hotel management (Admin only)
router.post('/:id/hotels', authorize('admin'), addHotel);
router.put('/:id/hotels/:hotelId', authorize('admin'), updateHotel);
router.delete('/:id/hotels/:hotelId', authorize('admin'), deleteHotel);

// Flight management (Admin only)
router.post('/:id/flights', authorize('admin'), addFlight);
router.delete('/:id/flights/:flightId', authorize('admin'), deleteFlight);

module.exports = router;
