import api from './api';

// Run scenario simulation
export const runScenario = async (scenarioData) => {
  const response = await api.post('/scenarios/simulate', scenarioData);
  return response.data;
};

// Get scenario templates
export const getScenarioTemplates = async () => {
  const response = await api.get('/scenarios/templates');
  return response.data;
};

// Run what-if analysis
export const runWhatIfAnalysis = async (analysisData) => {
  const response = await api.post('/scenarios/what-if', analysisData);
  return response.data;
};

export default {
  runScenario,
  getScenarioTemplates,
  runWhatIfAnalysis
};
