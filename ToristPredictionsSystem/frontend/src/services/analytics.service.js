import api from './api';

// Get feature importance
export const getFeatureImportance = async (modelType = 'all') => {
  const response = await api.get('/analytics/feature-importance', {
    params: { modelType }
  });
  return response.data;
};

// Compare models
export const compareModels = async () => {
  const response = await api.get('/analytics/models/compare');
  return response.data;
};

// Get model metrics
export const getModelMetrics = async (model = 'all') => {
  const response = await api.get('/analytics/models/metrics', {
    params: { model }
  });
  return response.data;
};

// Get seasonal patterns
export const getSeasonalPatterns = async () => {
  const response = await api.get('/analytics/seasonal-patterns');
  return response.data;
};

// Get prediction accuracy tracking
export const getPredictionAccuracy = async () => {
  const response = await api.get('/analytics/prediction-accuracy');
  return response.data;
};

// Get 12-month forecast
export const getForecast = async (startYear, startMonth, model = 'rf') => {
  const response = await api.get('/analytics/forecast', {
    params: { startYear, startMonth, model }
  });
  return response.data;
};

// Get yearly trends
export const getYearlyTrends = async () => {
  const response = await api.get('/analytics/yearly-trends');
  return response.data;
};

// Get tourism dashboard with destinations
export const getTourismDashboard = async () => {
  const response = await api.get('/analytics/tourism-dashboard');
  return response.data;
};

// Get arrivals breakdown
export const getArrivalsBreakdown = async (type = 'category') => {
  const response = await api.get('/analytics/arrivals-breakdown', {
    params: { type }
  });
  return response.data;
};

export default {
  getFeatureImportance,
  compareModels,
  getModelMetrics,
  getSeasonalPatterns,
  getPredictionAccuracy,
  getForecast,
  getYearlyTrends,
  getTourismDashboard,
  getArrivalsBreakdown
};
