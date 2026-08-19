import { useState } from 'react'
import './PredictionForm.css'

function BatchPredictionForm({ onSubmit, loading }) {
  const currentYear = new Date().getFullYear()
  const currentMonth = new Date().getMonth() + 1

  const [formData, setFormData] = useState({
    startYear: currentYear,
    startMonth: currentMonth,
    endYear: currentYear,
    endMonth: currentMonth + 3 > 12 ? currentMonth + 3 - 12 : currentMonth + 3,
    dollarRate: ''
  })

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ]

  const years = Array.from({ length: 11 }, (_, i) => currentYear - 5 + i)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: name === 'dollarRate' ? value : parseInt(value)
    }))
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    
    // Validate date range
    const startDate = new Date(formData.startYear, formData.startMonth - 1)
    const endDate = new Date(formData.endYear, formData.endMonth - 1)
    
    if (endDate < startDate) {
      alert('End date must be after start date')
      return
    }

    const submissionData = {
      ...formData,
      dollarRate: parseFloat(formData.dollarRate)
    }
    
    onSubmit(submissionData)
  }

  const calculateMonthCount = () => {
    const start = new Date(formData.startYear, formData.startMonth - 1)
    const end = new Date(formData.endYear, formData.endMonth - 1)
    const months = (end.getFullYear() - start.getFullYear()) * 12 + (end.getMonth() - start.getMonth()) + 1
    return months > 0 ? months : 0
  }

  return (
    <div className="prediction-form-container">
      <form onSubmit={handleSubmit} className="prediction-form batch-form">
        {/* Form Header */}
        <div className="form-header">
          <h2 className="form-title">Create Batch Predictions</h2>
          <p className="form-description">
            Generate predictions for multiple months at once
          </p>
          <div className="batch-info-badge">
            <span className="badge-icon">📊</span>
            <span>{calculateMonthCount()} months selected</span>
          </div>
        </div>

        {/* Start Date */}
        <div className="form-section">
          <label className="section-label">
            <span className="label-icon">🗓️</span>
            Start Date
          </label>
          <div className="form-row">
            <div className="form-group">
              <label className="input-label">Year</label>
              <div className="select-wrapper">
                <select
                  name="startYear"
                  value={formData.startYear}
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
                  name="startMonth"
                  value={formData.startMonth}
                  onChange={handleChange}
                  className="form-select"
                  required
                >
                  {months.map((month, index) => (
                    <option key={index} value={index + 1}>
                      {month}
                    </option>
                  ))}
                </select>
                <span className="select-arrow">▼</span>
              </div>
            </div>
          </div>
        </div>

        {/* End Date */}
        <div className="form-section">
          <label className="section-label">
            <span className="label-icon">🏁</span>
            End Date
          </label>
          <div className="form-row">
            <div className="form-group">
              <label className="input-label">Year</label>
              <div className="select-wrapper">
                <select
                  name="endYear"
                  value={formData.endYear}
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
                  name="endMonth"
                  value={formData.endMonth}
                  onChange={handleChange}
                  className="form-select"
                  required
                >
                  {months.map((month, index) => (
                    <option key={index} value={index + 1}>
                      {month}
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
                placeholder="Enter dollar rate (e.g., 295.50)"
                step="0.01"
                min="0"
                required
              />
            </div>
            <p className="input-hint">
              This rate will be used for all predictions in the batch
            </p>
          </div>
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
                <span>Generating {calculateMonthCount()} Predictions...</span>
              </>
            ) : (
              <>
                <span className="button-icon">🚀</span>
                <span>Generate Batch Predictions</span>
              </>
            )}
          </button>
        </div>

        {/* Info Box */}
        <div className="info-box">
          <div className="info-icon">💡</div>
          <div className="info-content">
            <h4 className="info-title">Batch Prediction Benefits</h4>
            <p className="info-text">
              Generate multiple predictions at once to analyze trends over time. 
              Perfect for long-term planning and forecasting.
            </p>
          </div>
        </div>
      </form>
    </div>
  )
}

export default BatchPredictionForm