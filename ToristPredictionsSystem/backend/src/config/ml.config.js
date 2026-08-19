const path = require('path');

// Use venv Python if available, otherwise fall back to system Python
const projectRoot = path.join(__dirname, '../../..');
const venvPythonPath = path.join(projectRoot, '.venv', 'Scripts', 'python.exe');
const fs = require('fs');
const pythonPath = fs.existsSync(venvPythonPath) ? venvPythonPath : (process.env.PYTHON_PATH || 'python');

module.exports = {
  pythonPath: pythonPath,
  mlModelPath: path.join(__dirname, '../../..', 'Ml', 'T'),
  scriptsPath: path.join(__dirname, '../../..', 'Ml', 'T'),
  
  models: {
    rf_arrivals: 'rf_arrivals.pkl',
    xgb_arrivals: 'xgb_arrivals.pkl',
    rf_revenue: 'rf_revenue.pkl',
    xgb_revenue: 'xgb_revenue.pkl',
    rf_occupancy: 'rf_occupancy.pkl',
    xgb_occupancy: 'xgb_occupancy.pkl'
  },
  
  predictionTypes: {
    ARRIVALS: 'arrivals',
    REVENUE: 'revenue',
    OCCUPANCY: 'occupancy',
    ALL: 'all'
  },

  modelChoices: {
    RF: 'rf',
    XGB: 'xgb'
  },
  
  sriLankanMonths: [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ],

  // Default values for features (based on Sri Lanka averages)
  defaultFeatures: {
    dollarRate: 320,
    apparentTemperature: 28.0,
    sunshine: 6.0,
    rain: 100.0,
    precipitationHours: 10.0,
    numEstablishments: 2000,
    numRooms: 40000,
    airfareIndex: 100.0,
    cpi: 200.0,
    arrivalsLag1: 150000,
    arrivalsLag2: 145000,
    arrivalsLag3: 140000,
    arrivalsLag12: 150000,
    arrivalsRoll3: 145000,
    arrivalsRoll6: 142000,
    arrivalsStd3: 5000,
    arrivalsYoy: 0.05,
    revenueLag1: 0,
    hotelOccupancyRate: 0.65
  }
};