import { useState } from 'react'
import './PredictionForm.css'

function PredictionForm({ onSubmit, loading }) {
  const currentYear = new Date().getFullYear()
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [formData, setFormData] = useState({
    year: currentYear,
    month: new Date().getMonth() + 1,
    dollarRate: '320',
    predictionType: 'all',
    modelChoice: 'rf',
    // Advanced features with sensible defaults
    apparentTemperature: '28',
    sunshine: '6',
    rain: '100',
    precipitationHours: '10',
    numEstablishments: '2000',
    numRooms: '40000',
    airfareIndex: '100',
    cpi: '200',
    arrivalsLag1: '150000',
    arrivalsLag2: '145000',
    arrivalsLag3: '140000',
    arrivalsLag12: '150000',
    arrivalsRoll3: '145000',
    arrivalsRoll6: '142000',
    arrivalsStd3: '5000',
    arrivalsYoy: '0.05',
    revenueLag1: '0',
    hotelOccupancyRate: '0.65'
  })

  const months = [
    { value: 1, label: 'January' },
    { value: 2, label: 'February' },
    { value: 3, label: 'March' },
    { value: 4, label: 'April' },
    { value: 5, label: 'May' },
    { value: 6, label: 'June' },
    { value: 7, label: 'July' },
    { value: 8, label: 'August' },
    { value: 9, label: 'September' },
    { value: 10, label: 'October' },
    { value: 11, label: 'November' },
    { value: 12, label: 'December' }
  ]

  const predictionTypes = [
    { value: 'all', label: 'All Predictions', icon: '🌐', description: 'Complete analysis' },
    { value: 'tourist_arrivals', label: 'Tourist Arrivals', icon: '✈️', description: 'Arrival forecasts' },
    { value: 'revenue', label: 'Revenue', icon: '💰', description: 'Revenue projections' },
    { value: 'rooms', label: 'Occupancy', icon: '🏨', description: 'Hotel occupancy rate' }
  ]

  const modelChoices = [
    { value: 'rf', label: 'Random Forest', icon: '🌲', description: 'Stable & reliable' },
    { value: 'xgb', label: 'XGBoost', icon: '⚡', description: 'High performance' }
  ]

  const years = Array.from({ length: 11 }, (_, i) => currentYear - 5 + i)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'year' || name === 'month' ? parseInt(value) : value
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    const submissionData = {
      year: formData.year,
      month: formData.month,
      dollarRate: parseFloat(formData.dollarRate) || 320,
      predictionType: formData.predictionType,
      modelChoice: formData.modelChoice,
      // Include advanced features
      apparentTemperature: parseFloat(formData.apparentTemperature) || 28,
      sunshine: parseFloat(formData.sunshine) || 6,
      rain: parseFloat(formData.rain) || 100,
      precipitationHours: parseFloat(formData.precipitationHours) || 10,
      numEstablishments: parseFloat(formData.numEstablishments) || 2000,
      numRooms: parseFloat(formData.numRooms) || 40000,
      airfareIndex: parseFloat(formData.airfareIndex) || 100,
      cpi: parseFloat(formData.cpi) || 200,
      arrivalsLag1: parseFloat(formData.arrivalsLag1) || 150000,
      arrivalsLag2: parseFloat(formData.arrivalsLag2) || 145000,
      arrivalsLag3: parseFloat(formData.arrivalsLag3) || 140000,
      arrivalsLag12: parseFloat(formData.arrivalsLag12) || 150000,
      arrivalsRoll3: parseFloat(formData.arrivalsRoll3) || 145000,
      arrivalsRoll6: parseFloat(formData.arrivalsRoll6) || 142000,
      arrivalsStd3: parseFloat(formData.arrivalsStd3) || 5000,
      arrivalsYoy: parseFloat(formData.arrivalsYoy) || 0.05,
      revenueLag1: parseFloat(formData.revenueLag1) || 0,
      hotelOccupancyRate: parseFloat(formData.hotelOccupancyRate) || 0.65
    }
    onSubmit(submissionData)
  }

  return (
    <div className="prediction-form-container">
      <form onSubmit={handleSubmit} className="prediction-form">
        {/* Form Header */}
        <div className="form-header">
          <h2 className="form-title">Create New Prediction</h2>
          <p className="form-description">
            Fill in the details below to generate your prediction
          </p>
        </div>

        {/* Prediction Type Selection */}
        <div className="form-section">
          <label className="section-label">
            <span className="label-icon">🎯</span>
            Prediction Type
          </label>
          <div className="prediction-type-grid">
            {predictionTypes.map((type) => (
              <label
                key={type.value}
                className={`type-card ${formData.predictionType === type.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="predictionType"
                  value={type.value}
                  checked={formData.predictionType === type.value}
                  onChange={handleChange}
                  className="type-radio"
                />
                <div className="type-icon">{type.icon}</div>
                <div className="type-label">{type.label}</div>
                <div className="type-description">{type.description}</div>
                <div className="type-check">✓</div>
              </label>
            ))}
          </div>
        </div>

        {/* Model Selection */}
        <div className="form-section">
          <label className="section-label">
            <span className="label-icon">🤖</span>
            Model Type
          </label>
          <div className="model-choice-grid">
            {modelChoices.map((model) => (
              <label
                key={model.value}
                className={`type-card model-card ${formData.modelChoice === model.value ? 'selected' : ''}`}
              >
                <input
                  type="radio"
                  name="modelChoice"
                  value={model.value}
                  checked={formData.modelChoice === model.value}
                  onChange={handleChange}
                  className="type-radio"
                />
                <div className="type-icon">{model.icon}</div>
                <div className="type-label">{model.label}</div>
                <div className="type-description">{model.description}</div>
                <div className="type-check">✓</div>
              </label>
            ))}
          </div>
        </div>

        {/* Date Selection */}
        <div className="form-section">
          <label className="section-label">
            <span className="label-icon">📅</span>
            Select Date
          </label>
          <div className="form-row">
            <div className="form-group">
              <label className="input-label">Year</label>
              <div className="select-wrapper">
                <select
                  name="year"
                  value={formData.year}
                  onChange={handleChange}
                  className="form-select"
                  required
                >
                  {years.map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
                <span className="select-arrow">▼</span>
              </div>
            </div>

            <div className="form-group">
              <label className="input-label">Month</label>
              <div className="select-wrapper">
                <select
                  name="month"
                  value={formData.month}
                  onChange={handleChange}
                  className="form-select"
                  required
                >
                  {months.map((month) => (
                    <option key={month.value} value={month.value}>
                      {month.label}
                    </option>
                  ))}
                </select>
                <span className="select-arrow">▼</span>
              </div>
            </div>
          </div>
        </div>

        {/* Dollar Rate */}
        <div className="form-section">
          <label className="section-label">
            <span className="label-icon">💵</span>
            Dollar Exchange Rate
          </label>
          <div className="form-group">
            <div className="input-wrapper">
              <span className="input-prefix">$</span>
              <input
                type="number"
                name="dollarRate"
                value={formData.dollarRate}
                onChange={handleChange}
                className="form-input"
                placeholder="Enter dollar rate (e.g., 320)"
                step="0.01"
                min="0"
                required
              />
            </div>
            <p className="input-hint">
              Enter the current USD to LKR exchange rate
            </p>
          </div>
        </div>

        {/* Advanced Options Toggle */}
        <div className="form-section">
          <button
            type="button"
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <span className="toggle-icon">{showAdvanced ? '▼' : '▶'}</span>
            <span className="toggle-label">Advanced Options</span>
            <span className="toggle-hint">(Optional)</span>
          </button>

          {showAdvanced && (
            <div className="advanced-options">
              {/* Weather Conditions */}
              <div className="advanced-group">
                <h4 className="advanced-group-title">🌤️ Weather Conditions</h4>
                <div className="advanced-grid">
                  <div className="form-group compact">
                    <label className="input-label">Temperature (°C)</label>
                    <input
                      type="number"
                      name="apparentTemperature"
                      value={formData.apparentTemperature}
                      onChange={handleChange}
                      className="form-input compact"
                      step="0.1"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">Sunshine (hrs)</label>
                    <input
                      type="number"
                      name="sunshine"
                      value={formData.sunshine}
                      onChange={handleChange}
                      className="form-input compact"
                      step="0.1"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">Rain (mm)</label>
                    <input
                      type="number"
                      name="rain"
                      value={formData.rain}
                      onChange={handleChange}
                      className="form-input compact"
                      step="1"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">Precipitation (hrs)</label>
                    <input
                      type="number"
                      name="precipitationHours"
                      value={formData.precipitationHours}
                      onChange={handleChange}
                      className="form-input compact"
                      step="0.1"
                    />
                  </div>
                </div>
              </div>

              {/* Infrastructure */}
              <div className="advanced-group">
                <h4 className="advanced-group-title">🏨 Infrastructure</h4>
                <div className="advanced-grid">
                  <div className="form-group compact">
                    <label className="input-label">Establishments</label>
                    <input
                      type="number"
                      name="numEstablishments"
                      value={formData.numEstablishments}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">Total Rooms</label>
                    <input
                      type="number"
                      name="numRooms"
                      value={formData.numRooms}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">Occupancy Rate</label>
                    <input
                      type="number"
                      name="hotelOccupancyRate"
                      value={formData.hotelOccupancyRate}
                      onChange={handleChange}
                      className="form-input compact"
                      step="0.01"
                      min="0"
                      max="1"
                    />
                  </div>
                </div>
              </div>

              {/* Economic Indicators */}
              <div className="advanced-group">
                <h4 className="advanced-group-title">📈 Economic Indicators</h4>
                <div className="advanced-grid">
                  <div className="form-group compact">
                    <label className="input-label">Airfare Index</label>
                    <input
                      type="number"
                      name="airfareIndex"
                      value={formData.airfareIndex}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">CPI</label>
                    <input
                      type="number"
                      name="cpi"
                      value={formData.cpi}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                </div>
              </div>

              {/* Historical Arrivals (Lag Features) */}
              <div className="advanced-group">
                <h4 className="advanced-group-title">📊 Historical Arrivals Data</h4>
                <div className="advanced-grid">
                  <div className="form-group compact">
                    <label className="input-label">Last Month</label>
                    <input
                      type="number"
                      name="arrivalsLag1"
                      value={formData.arrivalsLag1}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">2 Months Ago</label>
                    <input
                      type="number"
                      name="arrivalsLag2"
                      value={formData.arrivalsLag2}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">3 Months Ago</label>
                    <input
                      type="number"
                      name="arrivalsLag3"
                      value={formData.arrivalsLag3}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">Last Year Same Month</label>
                    <input
                      type="number"
                      name="arrivalsLag12"
                      value={formData.arrivalsLag12}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                </div>
              </div>

              {/* Rolling Averages */}
              <div className="advanced-group">
                <h4 className="advanced-group-title">📉 Rolling Statistics</h4>
                <div className="advanced-grid">
                  <div className="form-group compact">
                    <label className="input-label">3-Month Avg</label>
                    <input
                      type="number"
                      name="arrivalsRoll3"
                      value={formData.arrivalsRoll3}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">6-Month Avg</label>
                    <input
                      type="number"
                      name="arrivalsRoll6"
                      value={formData.arrivalsRoll6}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">3-Month Std Dev</label>
                    <input
                      type="number"
                      name="arrivalsStd3"
                      value={formData.arrivalsStd3}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                  <div className="form-group compact">
                    <label className="input-label">YoY Growth</label>
                    <input
                      type="number"
                      name="arrivalsYoy"
                      value={formData.arrivalsYoy}
                      onChange={handleChange}
                      className="form-input compact"
                      step="0.01"
                    />
                  </div>
                </div>
              </div>

              {/* Revenue Lag (for revenue prediction) */}
              <div className="advanced-group">
                <h4 className="advanced-group-title">💵 Revenue History</h4>
                <div className="advanced-grid">
                  <div className="form-group compact">
                    <label className="input-label">Last Month Revenue</label>
                    <input
                      type="number"
                      name="revenueLag1"
                      value={formData.revenueLag1}
                      onChange={handleChange}
                      className="form-input compact"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Submit Button */}
        <div className="form-actions">
          <button
            type="submit"
            className="submit-button"
            disabled={loading}
          >
            {loading ? (
              <>
                <span className="button-spinner"></span>
                <span>Generating Prediction...</span>
              </>
            ) : (
              <>
                <span className="button-icon">🚀</span>
                <span>Generate Prediction</span>
              </>
            )}
          </button>
        </div>

        {/* Info Box */}
        <div className="info-box">
          <div className="info-icon">💡</div>
          <div className="info-content">
            <h4 className="info-title">How it works</h4>
            <p className="info-text">
              Our AI model analyzes historical data and current market conditions 
              to provide accurate predictions with confidence scores.
            </p>
          </div>
        </div>
      </form>
    </div>
  )
}

export default PredictionForm