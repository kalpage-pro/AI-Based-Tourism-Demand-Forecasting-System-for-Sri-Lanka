import api from './api';

// Export predictions to CSV
export const exportPredictionsCSV = async (filters = {}) => {
  const response = await api.get('/export/predictions/csv', {
    params: filters,
    responseType: 'blob'
  });
  
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `predictions_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
  
  return { success: true };
};

// Export predictions to PDF
export const exportPredictionsPDF = async (filters = {}) => {
  const response = await api.get('/export/predictions/pdf', {
    params: filters,
    responseType: 'blob'
  });
  
  // Create download link
  const url = window.URL.createObjectURL(new Blob([response.data], { type: 'application/pdf' }));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', `tourism_report_${Date.now()}.pdf`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
  
  return { success: true };
};

// Export historical data
export const exportHistoricalData = async (format = 'csv', filters = {}) => {
  const response = await api.get('/export/historical', {
    params: { format, ...filters },
    responseType: format === 'csv' ? 'blob' : 'json'
  });
  
  if (format === 'csv') {
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `historical_data_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  }
  
  return response.data;
};

// Get export history
export const getExportHistory = async () => {
  const response = await api.get('/export/history');
  return response.data;
};

export default {
  exportPredictionsCSV,
  exportPredictionsPDF,
  exportHistoricalData,
  getExportHistory
};
