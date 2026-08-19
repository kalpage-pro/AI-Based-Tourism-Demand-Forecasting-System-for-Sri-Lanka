import api from './api';

const adminService = {
  // File management
  uploadFile: async (formData) => {
    const response = await api.post('/admin/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getAllFiles: async () => {
    const response = await api.get('/admin/files');
    return response.data;
  },

  getAvailableFiles: async () => {
    const response = await api.get('/admin/files/available');
    return response.data;
  },

  downloadFile: async (fileId, filename) => {
    const response = await api.get(`/admin/files/download/${fileId}`, {
      responseType: 'blob',
    });
    
    // Create download link
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  },

  deleteFile: async (fileId) => {
    const response = await api.delete(`/admin/files/${fileId}`);
    return response.data;
  },

  toggleFileStatus: async (fileId) => {
    const response = await api.patch(`/admin/files/${fileId}/toggle`);
    return response.data;
  },

  // User management
  getAllUsers: async () => {
    const response = await api.get('/admin/users');
    return response.data;
  },

  getUser: async (userId) => {
    const response = await api.get(`/admin/users/${userId}`);
    return response.data;
  },

  createUser: async (userData) => {
    const response = await api.post('/admin/users', userData);
    return response.data;
  },

  updateUser: async (userId, userData) => {
    const response = await api.put(`/admin/users/${userId}`, userData);
    return response.data;
  },

  deleteUser: async (userId) => {
    const response = await api.delete(`/admin/users/${userId}`);
    return response.data;
  },

  updateUserRole: async (userId, role) => {
    const response = await api.patch(`/admin/users/${userId}/role`, { role });
    return response.data;
  },

  // CSV Prediction
  uploadCSVForPrediction: async (formData) => {
    const response = await api.post('/admin/csv-predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // Prediction management
  getAllPredictions: async (page = 1, limit = 20) => {
    const response = await api.get(`/admin/predictions?page=${page}&limit=${limit}`);
    return response.data;
  },

  deletePrediction: async (predictionId) => {
    const response = await api.delete(`/admin/predictions/${predictionId}`);
    return response.data;
  },

  // Dashboard stats
  getAdminStats: async () => {
    const response = await api.get('/admin/stats');
    return response.data;
  },
};

export default adminService;
