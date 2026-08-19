import api from './api';

class PredictionService {
  async createPrediction(predictionData) {
    try {
      const response = await api.post('/predictions', predictionData);
      return response.data;
    } catch (error) {
      console.error('Create prediction error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getPredictions() {
    try {
      const response = await api.get('/predictions');
      return response.data;
    } catch (error) {
      console.error('Get predictions error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getPredictionHistory() {
    try {
      const response = await api.get('/predictions');
      console.log('Prediction history response:', response.data);
      return response.data.data || [];
    } catch (error) {
      console.error('Get predictions error:', error.response?.data || error.message);
      throw error;
    }
  }

  async getPredictionById(id) {
    try {
      const response = await api.get(`/predictions/${id}`);
      return response.data.data;
    } catch (error) {
      console.error('Get prediction error:', error.response?.data || error.message);
      throw error;
    }
  }

  async batchPredictions(data) {
    try {
      const response = await api.post('/predictions/batch', data);
      return response.data;
    } catch (error) {
      console.error('Batch predictions error:', error.response?.data || error.message);
      throw error;
    }
  }
}

export default new PredictionService();

// Named exports for compatibility
export const createPrediction = async (predictionData) => {
  const service = new PredictionService();
  return service.createPrediction(predictionData);
};

export const getPredictionHistory = async () => {
  const service = new PredictionService();
  return service.getPredictionHistory();
};

export const getPredictionById = async (id) => {
  const service = new PredictionService();
  return service.getPredictionById(id);
};

export const deletePrediction = async (id) => {
  try {
    const response = await api.delete(`/predictions/${id}`);
    return response.data;
  } catch (error) {
    console.error('Delete prediction error:', error.response?.data || error.message);
    throw error;
  }
};