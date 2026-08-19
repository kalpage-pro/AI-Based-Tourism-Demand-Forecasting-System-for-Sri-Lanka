import api from './api';

// Get all destinations
export const getAllDestinations = async (params = {}) => {
  const response = await api.get('/destinations', { params });
  return response.data;
};

// Get single destination
export const getDestination = async (id) => {
  const response = await api.get(`/destinations/${id}`);
  return response.data;
};

// Get featured destinations
export const getFeaturedDestinations = async () => {
  const response = await api.get('/destinations/featured');
  return response.data;
};

// Get destination analytics
export const getDestinationAnalytics = async () => {
  const response = await api.get('/destinations/analytics');
  return response.data;
};

// Get economical flights
export const getEconomicalFlights = async () => {
  const response = await api.get('/destinations/economical-flights');
  return response.data;
};

// Get best hotels
export const getBestHotels = async (params = {}) => {
  const response = await api.get('/destinations/best-hotels', { params });
  return response.data;
};

// Create destination (Admin only)
export const createDestination = async (data) => {
  const response = await api.post('/destinations', data);
  return response.data;
};

// Update destination (Admin only)
export const updateDestination = async (id, data) => {
  const response = await api.put(`/destinations/${id}`, data);
  return response.data;
};

// Delete destination (Admin only)
export const deleteDestination = async (id) => {
  const response = await api.delete(`/destinations/${id}`);
  return response.data;
};

// Toggle featured status (Admin only)
export const toggleFeatured = async (id) => {
  const response = await api.patch(`/destinations/${id}/toggle-featured`);
  return response.data;
};

// Add hotel to destination (Admin only)
export const addHotel = async (destinationId, hotelData) => {
  const response = await api.post(`/destinations/${destinationId}/hotels`, hotelData);
  return response.data;
};

// Update hotel (Admin only)
export const updateHotel = async (destinationId, hotelId, hotelData) => {
  const response = await api.put(`/destinations/${destinationId}/hotels/${hotelId}`, hotelData);
  return response.data;
};

// Delete hotel (Admin only)
export const deleteHotel = async (destinationId, hotelId) => {
  const response = await api.delete(`/destinations/${destinationId}/hotels/${hotelId}`);
  return response.data;
};

// Add flight to destination (Admin only)
export const addFlight = async (destinationId, flightData) => {
  const response = await api.post(`/destinations/${destinationId}/flights`, flightData);
  return response.data;
};

// Delete flight (Admin only)
export const deleteFlight = async (destinationId, flightId) => {
  const response = await api.delete(`/destinations/${destinationId}/flights/${flightId}`);
  return response.data;
};

export default {
  getAllDestinations,
  getDestination,
  getFeaturedDestinations,
  getDestinationAnalytics,
  getEconomicalFlights,
  getBestHotels,
  createDestination,
  updateDestination,
  deleteDestination,
  toggleFeatured,
  addHotel,
  updateHotel,
  deleteHotel,
  addFlight,
  deleteFlight
};
