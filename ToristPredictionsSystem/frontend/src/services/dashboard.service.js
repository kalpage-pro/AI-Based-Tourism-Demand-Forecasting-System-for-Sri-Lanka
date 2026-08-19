import api from './api';

class DashboardService {
  async getStats() {
    try {
      const response = await api.get('/dashboard/stats');
      return response.data.data; // Extract data from response
    } catch (error) {
      console.error('Get stats error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getRecentPredictions(limit = 5) {
    try {
      const response = await api.get('/predictions', {
        params: { limit }
      });
      return response.data.data; // Extract data from response
    } catch (error) {
      console.error('Get recent predictions error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getTrends(year) {
    try {
      const response = await api.get('/dashboard/trends', {
        params: { year }
      });
      return response.data.data;
    } catch (error) {
      console.error('Get trends error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getComparison() {
    try {
      const response = await api.get('/dashboard/compare');
      return response.data.data;
    } catch (error) {
      console.error('Get comparison error:', error.response?.data || error.message);
      throw error;
    }
  }
}

export default new DashboardService();

// Named exports for compatibility
export const getDashboardStats = async () => {
  const service = new DashboardService();
  return service.getStats();
};

export const getRecentPredictions = async (limit = 5) => {
  const service = new DashboardService();
  return service.getRecentPredictions(limit);
};

export const getTrendData = async (period = '30d') => {
  const service = new DashboardService();
  return service.getTrends(period);
};