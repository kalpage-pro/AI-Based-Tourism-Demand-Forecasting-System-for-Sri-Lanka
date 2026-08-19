const { spawn } = require('child_process');
const path = require('path');
const mlConfig = require('../config/ml.config');

class MLService {
  constructor() {
    this.pythonPath = mlConfig.pythonPath;
    this.scriptsPath = mlConfig.scriptsPath;
    this.defaultFeatures = mlConfig.defaultFeatures;
  }

  // Make a single prediction with new models
  async predict({ 
    year, 
    month, 
    dollarRate,
    predictionType,
    modelChoice = 'rf',
    // Additional feature inputs (optional - will use defaults if not provided)
    apparentTemperature,
    sunshine,
    rain,
    precipitationHours,
    numEstablishments,
    numRooms,
    airfareIndex,
    cpi,
    arrivalsLag1,
    arrivalsLag2,
    arrivalsLag3,
    arrivalsLag12,
    arrivalsRoll3,
    arrivalsRoll6,
    arrivalsStd3,
    arrivalsYoy,
    revenueLag1,
    hotelOccupancyRate
  }) {
    return new Promise((resolve, reject) => {
      const startTime = Date.now();
      
      const pythonScript = path.join(this.scriptsPath, 'predict.py');
      
      // Map prediction type for backward compatibility
      let mappedType = predictionType;
      if (predictionType === 'tourist_arrivals') mappedType = 'arrivals';
      if (predictionType === 'rooms') mappedType = 'occupancy';
      
      const args = [
        pythonScript,
        '--year', year.toString(),
        '--month', month.toString(),
        '--dollar-rate', (dollarRate || this.defaultFeatures.dollarRate).toString(),
        '--type', mappedType || 'all',
        '--model', modelChoice || 'rf',
        '--apparent-temperature', (apparentTemperature || this.defaultFeatures.apparentTemperature).toString(),
        '--sunshine', (sunshine || this.defaultFeatures.sunshine).toString(),
        '--rain', (rain || this.defaultFeatures.rain).toString(),
        '--precipitation-hours', (precipitationHours || this.defaultFeatures.precipitationHours).toString(),
        '--num-establishments', (numEstablishments || this.defaultFeatures.numEstablishments).toString(),
        '--num-rooms', (numRooms || this.defaultFeatures.numRooms).toString(),
        '--airfare-index', (airfareIndex || this.defaultFeatures.airfareIndex).toString(),
        '--cpi', (cpi || this.defaultFeatures.cpi).toString(),
        '--arrivals-lag1', (arrivalsLag1 || this.defaultFeatures.arrivalsLag1).toString(),
        '--arrivals-lag2', (arrivalsLag2 || this.defaultFeatures.arrivalsLag2).toString(),
        '--arrivals-lag3', (arrivalsLag3 || this.defaultFeatures.arrivalsLag3).toString(),
        '--arrivals-lag12', (arrivalsLag12 || this.defaultFeatures.arrivalsLag12).toString(),
        '--arrivals-roll3', (arrivalsRoll3 || this.defaultFeatures.arrivalsRoll3).toString(),
        '--arrivals-roll6', (arrivalsRoll6 || this.defaultFeatures.arrivalsRoll6).toString(),
        '--arrivals-std3', (arrivalsStd3 || this.defaultFeatures.arrivalsStd3).toString(),
        '--arrivals-yoy', (arrivalsYoy || this.defaultFeatures.arrivalsYoy).toString(),
        '--revenue-lag1', (revenueLag1 || this.defaultFeatures.revenueLag1).toString(),
        '--hotel-occupancy-rate', (hotelOccupancyRate || this.defaultFeatures.hotelOccupancyRate).toString()
      ];

      const pythonProcess = spawn(this.pythonPath, args, {
        cwd: this.scriptsPath,
        env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
      });

      let dataString = '';
      let errorString = '';

      pythonProcess.stdout.on('data', (data) => {
        dataString += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        errorString += data.toString();
        console.error('Python stderr:', data.toString());
      });

      pythonProcess.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        reject(new Error(`Failed to start Python: ${err.message}`));
      });

      pythonProcess.on('close', (code) => {
        const processingTime = Date.now() - startTime;
        
        console.log('Python process closed with code:', code);
        console.log('Python stdout:', dataString);
        if (errorString) console.log('Python stderr:', errorString);

        if (code !== 0) {
          reject(new Error(`Python script failed: ${errorString || dataString || 'Unknown error'}`));
          return;
        }

        try {
          const result = JSON.parse(dataString);
          
          if (!result.success) {
            reject(new Error(result.error || 'Prediction failed'));
            return;
          }
          
          resolve({
            predictions: {
              touristArrivals: {
                value: Math.round(result.tourist_arrivals || 0),
                confidence: result.confidence_tourist_arrivals || 0.85
              },
              revenue: {
                value: Math.round(result.revenue || 0),
                currency: 'USD',
                confidence: result.confidence_revenue || 0.82
              },
              rooms: {
                value: Math.round(result.rooms || 0),
                confidence: result.confidence_rooms || 0.80
              },
              occupancy: {
                value: result.occupancy || 0,
                percentage: result.hotel_occupancy_rate || 0,
                confidence: result.confidence_occupancy || 0.80
              }
            },
            metadata: {
              modelVersion: result.model_version || '2.0.0',
              modelType: result.model_type || 'rf',
              processingTime,
              accuracy: result.overall_accuracy || 0.87
            }
          });
        } catch (error) {
          reject(new Error(`Failed to parse prediction result: ${error.message}`));
        }
      });
    });
  }

  // Batch predictions for multiple months
  async batchPredict({ startYear, startMonth, endYear, endMonth, dollarRate }) {
    const predictions = [];
    let currentYear = startYear;
    let currentMonth = startMonth;

    while (
      currentYear < endYear ||
      (currentYear === endYear && currentMonth <= endMonth)
    ) {
      try {
        const result = await this.predict({
          year: currentYear,
          month: currentMonth,
          dollarRate,
          predictionType: 'all'
        });

        predictions.push({
          inputData: {
            year: currentYear,
            month: currentMonth,
            dollarRate
          },
          predictions: result.predictions,
          metadata: result.metadata
        });
      } catch (error) {
        console.error(`Failed to predict for ${currentYear}-${currentMonth}:`, error.message);
      }

      // Move to next month
      currentMonth++;
      if (currentMonth > 12) {
        currentMonth = 1;
        currentYear++;
      }
    }

    return predictions;
  }

  // Train or retrain models
  async trainModels() {
    return new Promise((resolve, reject) => {
      const pythonScript = path.join(this.scriptsPath, 'train_model.py');
      const pythonProcess = spawn(this.pythonPath, [pythonScript]);

      let output = '';
      let errorOutput = '';

      pythonProcess.stdout.on('data', (data) => {
        output += data.toString();
        console.log('Training output:', data.toString());
      });

      pythonProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
        console.error('Training error:', data.toString());
      });

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Training failed: ${errorOutput}`));
          return;
        }

        resolve({
          success: true,
          message: 'Models trained successfully',
          output
        });
      });
    });
  }

  // Get model information
  getModelInfo() {
    return {
      models: mlConfig.models,
      modelPath: mlConfig.mlModelPath,
      availableTypes: mlConfig.predictionTypes,
      pythonVersion: this.pythonPath
    };
  }
}

module.exports = new MLService();