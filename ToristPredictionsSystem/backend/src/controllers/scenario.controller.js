const { spawn } = require('child_process');
const path = require('path');
const mlConfig = require('../config/ml.config');
const ActivityLog = require('../models/ActivityLog.model');

// Run scenario simulation
exports.runScenario = async (req, res, next) => {
  try {
    const {
      baseYear,
      baseMonth,
      scenarios,
      model = 'rf'
    } = req.body;

    // Validate input
    if (!baseYear || !baseMonth || !scenarios || !Array.isArray(scenarios)) {
      return res.status(400).json({
        success: false,
        message: 'Please provide baseYear, baseMonth, and scenarios array'
      });
    }

    const results = [];
    
    // Get baseline prediction first
    const baselineResult = await runPythonScenario({
      year: baseYear,
      month: baseMonth,
      model,
      dollarRate: 320,
      apparentTemperature: 27,
      cpi: 250
    });
    
    results.push({
      name: 'Baseline',
      description: 'Current conditions',
      ...baselineResult
    });

    // Run each scenario
    for (const scenario of scenarios) {
      const scenarioResult = await runPythonScenario({
        year: baseYear,
        month: baseMonth,
        model,
        dollarRate: scenario.dollarRate || 320,
        apparentTemperature: scenario.temperature || 27,
        cpi: scenario.cpi || 250,
        numRooms: scenario.numRooms,
        numEstablishments: scenario.numEstablishments,
        airfareIndex: scenario.airfareIndex
      });
      
      // Calculate impact vs baseline
      const arrivalsImpact = baselineResult.arrivals > 0 
        ? ((scenarioResult.arrivals - baselineResult.arrivals) / baselineResult.arrivals * 100).toFixed(2)
        : 0;
      const revenueImpact = baselineResult.revenue > 0
        ? ((scenarioResult.revenue - baselineResult.revenue) / baselineResult.revenue * 100).toFixed(2)
        : 0;
      
      results.push({
        name: scenario.name,
        description: scenario.description,
        parameters: {
          dollarRate: scenario.dollarRate,
          temperature: scenario.temperature,
          cpi: scenario.cpi,
          numRooms: scenario.numRooms,
          numEstablishments: scenario.numEstablishments,
          airfareIndex: scenario.airfareIndex
        },
        ...scenarioResult,
        impact: {
          arrivals: parseFloat(arrivalsImpact),
          revenue: parseFloat(revenueImpact)
        }
      });
    }

    // Log the activity
    await ActivityLog.log(req.user._id, 'SCENARIO_SIMULATION', {
      baseYear,
      baseMonth,
      scenariosCount: scenarios.length
    }, req);

    res.status(200).json({
      success: true,
      data: {
        baseYear,
        baseMonth,
        model,
        results
      }
    });
  } catch (error) {
    next(error);
  }
};

// Get predefined scenario templates
exports.getScenarioTemplates = async (req, res, next) => {
  try {
    const templates = [
      {
        id: 'economic_boom',
        name: 'Economic Boom',
        description: 'Strong economy with favorable exchange rate',
        parameters: {
          dollarRate: 280,
          cpi: 230,
          airfareIndex: 95
        }
      },
      {
        id: 'economic_crisis',
        name: 'Economic Crisis',
        description: 'Currency depreciation and high inflation',
        parameters: {
          dollarRate: 400,
          cpi: 320,
          airfareIndex: 130
        }
      },
      {
        id: 'infrastructure_boost',
        name: 'Infrastructure Boost',
        description: 'Increased hotel capacity',
        parameters: {
          numRooms: 45000,
          numEstablishments: 4500
        }
      },
      {
        id: 'climate_favorable',
        name: 'Favorable Climate',
        description: 'Ideal weather conditions',
        parameters: {
          temperature: 26,
          sunshine: 9,
          rain: 50
        }
      },
      {
        id: 'monsoon_heavy',
        name: 'Heavy Monsoon',
        description: 'Adverse weather conditions',
        parameters: {
          temperature: 28,
          sunshine: 4,
          rain: 350,
          precipitationHours: 180
        }
      },
      {
        id: 'tourism_campaign',
        name: 'Tourism Promotion Campaign',
        description: 'Successful marketing leading to 20% increase in lag arrivals',
        parameters: {
          arrivalsLag1: 180000,
          arrivalsLag2: 175000,
          arrivalsLag3: 170000
        }
      }
    ];

    res.status(200).json({
      success: true,
      data: templates
    });
  } catch (error) {
    next(error);
  }
};

// What-if analysis for single parameter
exports.whatIfAnalysis = async (req, res, next) => {
  try {
    const {
      parameter,
      minValue,
      maxValue,
      steps = 5,
      baseYear,
      baseMonth,
      model = 'rf'
    } = req.body;

    const stepSize = (maxValue - minValue) / (steps - 1);
    const results = [];

    for (let i = 0; i < steps; i++) {
      const value = minValue + (stepSize * i);
      const params = {
        year: baseYear,
        month: baseMonth,
        model
      };
      
      // Set the varying parameter
      params[parameter] = value;
      
      const prediction = await runPythonScenario(params);
      
      results.push({
        [parameter]: Math.round(value * 100) / 100,
        arrivals: prediction.arrivals,
        revenue: prediction.revenue,
        occupancy: prediction.occupancy
      });
    }

    res.status(200).json({
      success: true,
      data: {
        parameter,
        range: { min: minValue, max: maxValue },
        results
      }
    });
  } catch (error) {
    next(error);
  }
};

// Helper function to run Python prediction
function runPythonScenario(params) {
  return new Promise((resolve, reject) => {
    const args = [
      path.join(mlConfig.scriptsPath, 'predict.py'),
      '--year', (params.year || 2025).toString(),
      '--month', (params.month || 1).toString(),
      '--model', params.model || 'rf',
      '--type', 'all',
      '--dollar-rate', (params.dollarRate || 320).toString(),
      '--apparent-temperature', (params.apparentTemperature || 27).toString(),
      '--cpi', (params.cpi || 250).toString()
    ];

    if (params.numRooms) args.push('--num-rooms', params.numRooms.toString());
    if (params.numEstablishments) args.push('--num-establishments', params.numEstablishments.toString());
    if (params.airfareIndex) args.push('--airfare-index', params.airfareIndex.toString());
    if (params.arrivalsLag1) args.push('--arrivals-lag1', params.arrivalsLag1.toString());
    if (params.arrivalsLag2) args.push('--arrivals-lag2', params.arrivalsLag2.toString());
    if (params.arrivalsLag3) args.push('--arrivals-lag3', params.arrivalsLag3.toString());
    if (params.sunshine) args.push('--sunshine', params.sunshine.toString());
    if (params.rain) args.push('--rain', params.rain.toString());

    const pythonProcess = spawn(mlConfig.pythonPath, args, {
      cwd: mlConfig.scriptsPath,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' }
    });

    let dataString = '';
    let errorString = '';

    pythonProcess.stdout.on('data', (data) => {
      dataString += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorString += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(dataString);
          resolve({
            arrivals: result.arrivals || result.predictions?.arrivals?.value || 0,
            revenue: result.revenue || result.predictions?.revenue?.value || 0,
            occupancy: result.occupancy || result.predictions?.occupancy?.value || 0
          });
        } catch (e) {
          resolve({ arrivals: 0, revenue: 0, occupancy: 0 });
        }
      } else {
        reject(new Error(`Scenario simulation failed: ${errorString}`));
      }
    });
  });
}
