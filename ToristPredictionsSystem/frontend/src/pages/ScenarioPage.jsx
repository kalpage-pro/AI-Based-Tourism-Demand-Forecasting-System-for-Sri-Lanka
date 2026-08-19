import { useState, useEffect } from 'react';
import { runScenario, getScenarioTemplates, runWhatIfAnalysis } from '../services/scenario.service';
import Loading from '../components/Common/Loading';
import Card from '../components/Common/Card';
import Button from '../components/Common/Button';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, LineChart, Line
} from 'recharts';
import './ScenarioPage.css';

function ScenarioPage() {
  const [loading, setLoading] = useState(false);
  const [templates, setTemplates] = useState([]);
  const [selectedTemplates, setSelectedTemplates] = useState([]);
  const [baseYear, setBaseYear] = useState(new Date().getFullYear());
  const [baseMonth, setBaseMonth] = useState(new Date().getMonth() + 1);
  const [model, setModel] = useState('rf');
  const [results, setResults] = useState(null);
  const [whatIfResults, setWhatIfResults] = useState(null);
  const [whatIfParam, setWhatIfParam] = useState('dollarRate');
  const [whatIfMin, setWhatIfMin] = useState(280);
  const [whatIfMax, setWhatIfMax] = useState(400);
  const [activeTab, setActiveTab] = useState('scenario');
  const [error, setError] = useState(null);
  const [customScenario, setCustomScenario] = useState({
    name: 'Custom Scenario',
    description: 'My custom scenario',
    dollarRate: 320,
    temperature: 27,
    cpi: 250
  });

  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    try {
      const response = await getScenarioTemplates();
      console.log('Templates response:', response);
      setTemplates(response.data || []);
    } catch (err) {
      console.error('Failed to load templates:', err);
      setError('Failed to load scenario templates. Please make sure you are logged in.');
    }
  };

  const toggleTemplate = (templateId) => {
    setSelectedTemplates(prev => 
      prev.includes(templateId)
        ? prev.filter(id => id !== templateId)
        : [...prev, templateId]
    );
  };

  const runSimulation = async () => {
    try {
      setLoading(true);
      setError(null);

      const scenarios = selectedTemplates.map(id => {
        const template = templates.find(t => t.id === id);
        return {
          name: template.name,
          description: template.description,
          ...template.parameters
        };
      });

      // Add custom scenario if it has changes
      if (customScenario.name && customScenario.name !== 'Custom Scenario') {
        scenarios.push(customScenario);
      }

      if (scenarios.length === 0) {
        // Just run baseline if no scenarios selected
        scenarios.push({
          name: 'Test Scenario',
          description: 'Same as baseline',
          dollarRate: 320
        });
      }

      const response = await runScenario({
        baseYear,
        baseMonth,
        model,
        scenarios
      });

      console.log('Scenario response:', response);
      setResults(response.data);
    } catch (err) {
      setError('Failed to run simulation. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const runWhatIf = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await runWhatIfAnalysis({
        parameter: whatIfParam,
        minValue: parseFloat(whatIfMin),
        maxValue: parseFloat(whatIfMax),
        steps: 6,
        baseYear,
        baseMonth,
        model
      });

      console.log('What-If response:', response);
      setWhatIfResults(response.data);
    } catch (err) {
      setError('Failed to run what-if analysis. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getMonthName = (month) => {
    const months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December'];
    return months[month];
  };

  const parameterOptions = [
    { value: 'dollarRate', label: 'USD/LKR Exchange Rate', min: 280, max: 450 },
    { value: 'apparentTemperature', label: 'Temperature (°C)', min: 22, max: 34 },
    { value: 'cpi', label: 'Consumer Price Index', min: 180, max: 350 },
    { value: 'numRooms', label: 'Number of Hotel Rooms', min: 30000, max: 60000 },
    { value: 'airfareIndex', label: 'Airfare Index', min: 80, max: 150 }
  ];

  return (
    <div className="scenario-page">
      <div className="scenario-header">
        <h1>🎮 Scenario Simulation</h1>
        <p>Explore "what-if" scenarios and their impact on tourism predictions</p>
      </div>

      <div className="scenario-tabs">
        <button 
          className={`tab-btn ${activeTab === 'scenario' ? 'active' : ''}`}
          onClick={() => setActiveTab('scenario')}
        >
          📊 Scenario Comparison
        </button>
        <button 
          className={`tab-btn ${activeTab === 'whatif' ? 'active' : ''}`}
          onClick={() => setActiveTab('whatif')}
        >
          🔍 What-If Analysis
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {activeTab === 'scenario' && (
        <div className="scenario-content">
          <div className="scenario-builder">
            <Card className="config-card">
              <h3>⚙️ Simulation Settings</h3>
              
              <div className="settings-row">
                <div className="setting-group">
                  <label>Base Year</label>
                  <select value={baseYear} onChange={(e) => setBaseYear(parseInt(e.target.value))}>
                    {[2024, 2025, 2026, 2027, 2028].map(y => (
                      <option key={y} value={y}>{y}</option>
                    ))}
                  </select>
                </div>
                
                <div className="setting-group">
                  <label>Base Month</label>
                  <select value={baseMonth} onChange={(e) => setBaseMonth(parseInt(e.target.value))}>
                    {[1,2,3,4,5,6,7,8,9,10,11,12].map(m => (
                      <option key={m} value={m}>{getMonthName(m)}</option>
                    ))}
                  </select>
                </div>
                
                <div className="setting-group">
                  <label>Model</label>
                  <select value={model} onChange={(e) => setModel(e.target.value)}>
                    <option value="rf">🌲 Random Forest</option>
                    <option value="xgb">⚡ XGBoost</option>
                  </select>
                </div>
              </div>
            </Card>

            <Card className="templates-card">
              <h3>📋 Scenario Templates</h3>
              <p className="templates-hint">Select scenarios to compare against baseline</p>
              
              <div className="templates-grid">
                {templates.map(template => (
                  <div 
                    key={template.id}
                    className={`template-card ${selectedTemplates.includes(template.id) ? 'selected' : ''}`}
                    onClick={() => toggleTemplate(template.id)}
                  >
                    <div className="template-checkbox">
                      {selectedTemplates.includes(template.id) ? '✓' : ''}
                    </div>
                    <div className="template-info">
                      <h4>{template.name}</h4>
                      <p>{template.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </Card>

            <Card className="custom-scenario-card">
              <h3>✏️ Custom Scenario (Optional)</h3>
              
              <div className="custom-inputs">
                <div className="input-group">
                  <label>Scenario Name</label>
                  <input
                    type="text"
                    value={customScenario.name}
                    onChange={(e) => setCustomScenario({...customScenario, name: e.target.value})}
                    placeholder="My Custom Scenario"
                  />
                </div>
                
                <div className="input-row">
                  <div className="input-group">
                    <label>Dollar Rate (LKR)</label>
                    <input
                      type="number"
                      value={customScenario.dollarRate}
                      onChange={(e) => setCustomScenario({...customScenario, dollarRate: parseFloat(e.target.value)})}
                    />
                  </div>
                  
                  <div className="input-group">
                    <label>Temperature (°C)</label>
                    <input
                      type="number"
                      value={customScenario.temperature}
                      onChange={(e) => setCustomScenario({...customScenario, temperature: parseFloat(e.target.value)})}
                    />
                  </div>
                  
                  <div className="input-group">
                    <label>CPI</label>
                    <input
                      type="number"
                      value={customScenario.cpi}
                      onChange={(e) => setCustomScenario({...customScenario, cpi: parseFloat(e.target.value)})}
                    />
                  </div>
                </div>
              </div>
            </Card>

            <Button 
              onClick={runSimulation} 
              disabled={loading}
              className="run-btn"
            >
              {loading ? '⏳ Running Simulation...' : '🚀 Run Simulation'}
            </Button>
          </div>

          {results && (
            <div className="results-section">
              <Card className="results-card">
                <h3>📊 Simulation Results - {getMonthName(results.baseMonth)} {results.baseYear}</h3>
                
                {results.results && results.results.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={results.results} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis type="category" dataKey="name" width={150} />
                        <Tooltip formatter={(value) => value?.toLocaleString()} />
                        <Legend />
                        <Bar dataKey="arrivals" name="Tourist Arrivals" fill="#3b82f6" />
                      </BarChart>
                    </ResponsiveContainer>

                    <div className="results-table">
                      <table>
                        <thead>
                          <tr>
                            <th>Scenario</th>
                            <th>Arrivals</th>
                            <th>Revenue</th>
                            <th>Impact vs Baseline</th>
                          </tr>
                    </thead>
                    <tbody>
                      {results.results.map((r, idx) => (
                        <tr key={idx} className={r.name === 'Baseline' ? 'baseline-row' : ''}>
                          <td>
                            <strong>{r.name}</strong>
                            <span className="scenario-desc">{r.description}</span>
                          </td>
                          <td>{r.arrivals?.toLocaleString() || '-'}</td>
                          <td>${(r.revenue / 1000000)?.toFixed(2)}M</td>
                          <td>
                            {r.impact ? (
                              <span className={`impact ${r.impact.arrivals >= 0 ? 'positive' : 'negative'}`}>
                                {r.impact.arrivals >= 0 ? '+' : ''}{r.impact.arrivals}%
                              </span>
                            ) : (
                              <span className="baseline-badge">BASELINE</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                  </>
                ) : (
                  <div className="no-results">
                    <p>No results data available. Please try running the simulation again.</p>
                  </div>
                )}
              </Card>
            </div>
          )}
        </div>
      )}

      {activeTab === 'whatif' && (
        <div className="whatif-content">
          <Card className="whatif-config">
            <h3>🔍 What-If Parameter Analysis</h3>
            <p>See how a single parameter change affects predictions</p>
            
            <div className="whatif-inputs">
              <div className="input-group">
                <label>Parameter to Analyze</label>
                <select 
                  value={whatIfParam} 
                  onChange={(e) => {
                    const param = parameterOptions.find(p => p.value === e.target.value);
                    setWhatIfParam(e.target.value);
                    setWhatIfMin(param.min);
                    setWhatIfMax(param.max);
                  }}
                >
                  {parameterOptions.map(p => (
                    <option key={p.value} value={p.value}>{p.label}</option>
                  ))}
                </select>
              </div>
              
              <div className="input-group">
                <label>Min Value</label>
                <input
                  type="number"
                  value={whatIfMin}
                  onChange={(e) => setWhatIfMin(e.target.value)}
                />
              </div>
              
              <div className="input-group">
                <label>Max Value</label>
                <input
                  type="number"
                  value={whatIfMax}
                  onChange={(e) => setWhatIfMax(e.target.value)}
                />
              </div>

              <div className="input-group">
                <label>Model</label>
                <select value={model} onChange={(e) => setModel(e.target.value)}>
                  <option value="rf">🌲 Random Forest</option>
                  <option value="xgb">⚡ XGBoost</option>
                </select>
              </div>
            </div>
            
            <Button onClick={runWhatIf} disabled={loading} className="run-btn">
              {loading ? '⏳ Analyzing...' : '📈 Run Analysis'}
            </Button>
          </Card>

          {whatIfResults && (
            <Card className="whatif-results">
              <h3>
                📊 {parameterOptions.find(p => p.value === whatIfResults.parameter)?.label} Impact
              </h3>
              
              {whatIfResults.results && whatIfResults.results.length > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={whatIfResults.results}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey={whatIfResults.parameter} />
                      <YAxis />
                      <Tooltip formatter={(value) => value?.toLocaleString()} />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="arrivals" 
                        name="Tourist Arrivals"
                        stroke="#3b82f6" 
                        strokeWidth={3}
                        dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <div className="whatif-table">
                    <table>
                      <thead>
                        <tr>
                          <th>{parameterOptions.find(p => p.value === whatIfResults.parameter)?.label}</th>
                          <th>Arrivals</th>
                          <th>Revenue</th>
                        </tr>
                      </thead>
                      <tbody>
                        {whatIfResults.results.map((r, idx) => (
                          <tr key={idx}>
                            <td>{r[whatIfResults.parameter]}</td>
                            <td>{r.arrivals?.toLocaleString() || '-'}</td>
                            <td>${(r.revenue / 1000000)?.toFixed(2)}M</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="no-results">
                  <p>No analysis data available. Please try running the analysis again.</p>
                </div>
              )}
            </Card>
          )}
        </div>
      )}
    </div>
  );
}

export default ScenarioPage;
