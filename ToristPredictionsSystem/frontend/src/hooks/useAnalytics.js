import { useState, useCallback } from 'react';
import analyticsService from '../services/analytics.service';

export const useAnalytics = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState({
    featureImportance: null,
    modelComparison: null,
    seasonalPatterns: null,
    yearlyTrends: null,
    forecast: null,
    predictionAccuracy: null
  });

  const fetchFeatureImportance = useCallback(async (modelType = 'all') => {
    try {
      setLoading(true);
      const response = await analyticsService.getFeatureImportance(modelType);
      setData(prev => ({ ...prev, featureImportance: response.data }));
      return response.data;
    } catch (err) {
      setError('Failed to fetch feature importance');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchModelComparison = useCallback(async () => {
    try {
      setLoading(true);
      const response = await analyticsService.compareModels();
      setData(prev => ({ ...prev, modelComparison: response.data }));
      return response.data;
    } catch (err) {
      setError('Failed to fetch model comparison');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchSeasonalPatterns = useCallback(async () => {
    try {
      setLoading(true);
      const response = await analyticsService.getSeasonalPatterns();
      setData(prev => ({ ...prev, seasonalPatterns: response.data }));
      return response.data;
    } catch (err) {
      setError('Failed to fetch seasonal patterns');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchYearlyTrends = useCallback(async () => {
    try {
      setLoading(true);
      const response = await analyticsService.getYearlyTrends();
      setData(prev => ({ ...prev, yearlyTrends: response.data }));
      return response.data;
    } catch (err) {
      setError('Failed to fetch yearly trends');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchForecast = useCallback(async (startYear, startMonth, model = 'rf') => {
    try {
      setLoading(true);
      const response = await analyticsService.getForecast(startYear, startMonth, model);
      setData(prev => ({ ...prev, forecast: response.data }));
      return response.data;
    } catch (err) {
      setError('Failed to fetch forecast');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchPredictionAccuracy = useCallback(async () => {
    try {
      setLoading(true);
      const response = await analyticsService.getPredictionAccuracy();
      setData(prev => ({ ...prev, predictionAccuracy: response.data }));
      return response.data;
    } catch (err) {
      setError('Failed to fetch prediction accuracy');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAllAnalytics = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const [fi, mc, sp, yt, pa] = await Promise.all([
        analyticsService.getFeatureImportance('all').catch(() => ({ data: null })),
        analyticsService.compareModels().catch(() => ({ data: null })),
        analyticsService.getSeasonalPatterns().catch(() => ({ data: null })),
        analyticsService.getYearlyTrends().catch(() => ({ data: null })),
        analyticsService.getPredictionAccuracy().catch(() => ({ data: null }))
      ]);

      setData({
        featureImportance: fi.data,
        modelComparison: mc.data,
        seasonalPatterns: sp.data,
        yearlyTrends: yt.data,
        predictionAccuracy: pa.data,
        forecast: null
      });

      return data;
    } catch (err) {
      setError('Failed to fetch analytics data');
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    loading,
    error,
    data,
    fetchFeatureImportance,
    fetchModelComparison,
    fetchSeasonalPatterns,
    fetchYearlyTrends,
    fetchForecast,
    fetchPredictionAccuracy,
    fetchAllAnalytics,
    clearError
  };
};

export default useAnalytics;
