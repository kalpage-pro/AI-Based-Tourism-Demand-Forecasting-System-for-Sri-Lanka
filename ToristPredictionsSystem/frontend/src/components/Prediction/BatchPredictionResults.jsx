import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import './PredictionResult.css'

function BatchPredictionResults({ results, onNewPrediction }) {
  const navigate = useNavigate()
  const [selectedIndex, setSelectedIndex] = useState(0)

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-LK', {
      style: 'currency',
      currency: 'LKR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value)
  }

  const formatNumber = (value) => {
    return new Intl.NumberFormat('en-US').format(Math.round(value))
  }

  const formatDate = (year, month) => {
    const date = new Date(year, month - 1)
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
  }

  const selectedResult = results[selectedIndex]

  // Calculate totals
  const totals = results.reduce((acc, result) => ({
    arrivals: acc.arrivals + (result.predictions.touristArrivals?.value || 0),
    revenue: acc.revenue + (result.predictions.revenue?.value || 0),
    rooms: acc.rooms + (result.predictions.rooms?.value || 0)
  }), { arrivals: 0, revenue: 0, rooms: 0 })

  const averages = {
    arrivals: totals.arrivals / results.length,
    revenue: totals.revenue / results.length,
    rooms: totals.rooms / results.length
  }

  return (
    <div className="batch-results">
      {/* Success Header */}
      <div className="result-success-header">
        <div className="success-animation">
          <div className="success-checkmark">
            <div className="check-icon">
              <span className="icon-line line-tip"></span>
              <span className="icon-line line-long"></span>
              <div className="icon-circle"></div>
              <div className="icon-fix"></div>
            </div>
          </div>
        </div>
        <h2 className="success-title">Batch Predictions Complete!</h2>
        <p className="success-subtitle">
          Generated {results.length} predictions successfully
        </p>
      </div>

      {/* Summary Cards */}
      <div className="summary-grid">
        <div className="summary-card">
          <div className="summary-icon">✈️</div>
          <div className="summary-content">
            <div className="summary-label">Total Arrivals</div>
            <div className="summary-value">{formatNumber(totals.arrivals)}</div>
            <div className="summary-sublabel">Avg: {formatNumber(averages.arrivals)}/month</div>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">💰</div>
          <div className="summary-content">
            <div className="summary-label">Total Revenue</div>
            <div className="summary-value">{formatCurrency(totals.revenue)}</div>
            <div className="summary-sublabel">Avg: {formatCurrency(averages.revenue)}/month</div>
          </div>
        </div>

        <div className="summary-card">
          <div className="summary-icon">🏨</div>
          <div className="summary-content">
            <div className="summary-label">Total Rooms</div>
            <div className="summary-value">{formatNumber(totals.rooms)}</div>
            <div className="summary-sublabel">Avg: {formatNumber(averages.rooms)}/month</div>
          </div>
        </div>
      </div>

      {/* Timeline Navigator */}
      <div className="timeline-section">
        <h3 className="timeline-title">
          <span className="timeline-icon">📅</span>
          Monthly Breakdown
        </h3>
        <div className="timeline-navigator">
          {results.map((result, index) => (
            <button
              key={index}
              className={`timeline-item ${selectedIndex === index ? 'active' : ''}`}
              onClick={() => setSelectedIndex(index)}
            >
              <div className="timeline-date">
                {formatDate(result.inputData.year, result.inputData.month)}
              </div>
              <div className="timeline-value">
                {formatNumber(result.predictions.touristArrivals?.value || 0)}
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Selected Prediction Details */}
      {selectedResult && (
        <div className="selected-prediction">
          <h3 className="selected-title">
            {formatDate(selectedResult.inputData.year, selectedResult.inputData.month)} Details
          </h3>
          
          <div className="predictions-grid">
            {selectedResult.predictions.touristArrivals && (
              <div className="prediction-card arrivals-card">
                <div className="card-header">
                  <div className="card-icon-wrapper">
                    <span className="card-icon">✈️</span>
                  </div>
                  <h4 className="card-title">Tourist Arrivals</h4>
                </div>
                <div className="card-body">
                  <div className="prediction-value">
                    {formatNumber(selectedResult.predictions.touristArrivals.value)}
                  </div>
                  <div className="prediction-label">Expected Visitors</div>
                </div>
                <div className="card-footer">
                  <div className="confidence-badge">
                    <span className="confidence-icon">📊</span>
                    <span>{selectedResult.predictions.touristArrivals.confidence.toFixed(1)}% Confidence</span>
                  </div>
                </div>
              </div>
            )}

            {selectedResult.predictions.revenue && (
              <div className="prediction-card revenue-card">
                <div className="card-header">
                  <div className="card-icon-wrapper">
                    <span className="card-icon">💰</span>
                  </div>
                  <h4 className="card-title">Revenue</h4>
                </div>
                <div className="card-body">
                  <div className="prediction-value">
                    {formatCurrency(selectedResult.predictions.revenue.value)}
                  </div>
                  <div className="prediction-label">Projected Revenue</div>
                </div>
                <div className="card-footer">
                  <div className="confidence-badge">
                    <span className="confidence-icon">📊</span>
                    <span>{selectedResult.predictions.revenue.confidence.toFixed(1)}% Confidence</span>
                  </div>
                </div>
              </div>
            )}

            {selectedResult.predictions.rooms && (
              <div className="prediction-card rooms-card">
                <div className="card-header">
                  <div className="card-icon-wrapper">
                    <span className="card-icon">🏨</span>
                  </div>
                  <h4 className="card-title">Room Occupancy</h4>
                </div>
                <div className="card-body">
                  <div className="prediction-value">
                    {formatNumber(selectedResult.predictions.rooms.value)}
                  </div>
                  <div className="prediction-label">Expected Room Nights</div>
                </div>
                <div className="card-footer">
                  <div className="confidence-badge">
                    <span className="confidence-icon">📊</span>
                    <span>{selectedResult.predictions.rooms.confidence.toFixed(1)}% Confidence</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="result-actions">
        <button
          onClick={onNewPrediction}
          className="action-button primary-action"
        >
          <span className="button-icon">🎯</span>
          <span>Create New Batch</span>
        </button>
        <button
          onClick={() => navigate('/history')}
          className="action-button secondary-action"
        >
          <span className="button-icon">📋</span>
          <span>View All Predictions</span>
        </button>
      </div>
    </div>
  )
}

export default BatchPredictionResults