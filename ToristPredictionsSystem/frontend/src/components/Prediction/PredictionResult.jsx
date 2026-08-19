import { useNavigate } from 'react-router-dom'
import './PredictionResult.css'

function PredictionResult({ result, onNewPrediction }) {
  const navigate = useNavigate()

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
    return date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'high'
    if (confidence >= 75) return 'medium'
    return 'low'
  }

  const getPredictionTypeLabel = (type) => {
    const types = {
      'all': 'Complete Analysis',
      'tourist_arrivals': 'Tourist Arrivals',
      'revenue': 'Revenue Prediction',
      'rooms': 'Room Occupancy'
    }
    return types[type] || type
  }

  if (!result) return null

  return (
    <div className="prediction-result">
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
        <h2 className="success-title">Prediction Complete!</h2>
        <p className="success-subtitle">
          Your prediction has been generated successfully
        </p>
      </div>

      {/* Prediction Info Card */}
      <div className="result-info-card">
        <div className="info-row">
          <div className="info-item">
            <span className="info-icon">📅</span>
            <div className="info-details">
              <span className="info-label">Period</span>
              <span className="info-value">
                {formatDate(result.inputData.year, result.inputData.month)}
              </span>
            </div>
          </div>

          <div className="info-item">
            <span className="info-icon">💵</span>
            <div className="info-details">
              <span className="info-label">Exchange Rate</span>
              <span className="info-value">
                ${result.inputData.dollarRate?.toFixed(2) || '320.00'}
              </span>
            </div>
          </div>

          <div className="info-item">
            <span className="info-icon">🎯</span>
            <div className="info-details">
              <span className="info-label">Type</span>
              <span className="info-value">
                {getPredictionTypeLabel(result.predictionType)}
              </span>
            </div>
          </div>

          <div className="info-item">
            <span className="info-icon">🤖</span>
            <div className="info-details">
              <span className="info-label">Model</span>
              <span className="info-value">
                {result.inputData.modelChoice === 'xgb' ? 'XGBoost' : 'Random Forest'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Predictions Grid */}
      <div className="predictions-grid">
        {/* Tourist Arrivals */}
        {result.predictions.touristArrivals && (
          <div className="prediction-card arrivals-card">
            <div className="card-header">
              <div className="card-icon-wrapper">
                <span className="card-icon">✈️</span>
              </div>
              <h3 className="card-title">Tourist Arrivals</h3>
            </div>
            <div className="card-body">
              <div className="prediction-value">
                {formatNumber(result.predictions.touristArrivals.value)}
              </div>
              <div className="prediction-label">Expected Visitors</div>
            </div>
            <div className="card-footer">
              <div className={`confidence-badge confidence-${getConfidenceColor(result.predictions.touristArrivals.confidence)}`}>
                <span className="confidence-icon">📊</span>
                <span>{result.predictions.touristArrivals.confidence.toFixed(1)}% Confidence</span>
              </div>
            </div>
          </div>
        )}

        {/* Revenue */}
        {result.predictions.revenue && (
          <div className="prediction-card revenue-card">
            <div className="card-header">
              <div className="card-icon-wrapper">
                <span className="card-icon">💰</span>
              </div>
              <h3 className="card-title">Revenue</h3>
            </div>
            <div className="card-body">
              <div className="prediction-value">
                {formatCurrency(result.predictions.revenue.value)}
              </div>
              <div className="prediction-label">
                Projected Revenue ({result.predictions.revenue.currency})
              </div>
            </div>
            <div className="card-footer">
              <div className={`confidence-badge confidence-${getConfidenceColor(result.predictions.revenue.confidence)}`}>
                <span className="confidence-icon">📊</span>
                <span>{result.predictions.revenue.confidence.toFixed(1)}% Confidence</span>
              </div>
            </div>
          </div>
        )}

        {/* Occupancy/Rooms */}
        {result.predictions.rooms && (
          <div className="prediction-card rooms-card">
            <div className="card-header">
              <div className="card-icon-wrapper">
                <span className="card-icon">🏨</span>
              </div>
              <h3 className="card-title">Hotel Occupancy</h3>
            </div>
            <div className="card-body">
              {result.predictions.occupancy?.percentage ? (
                <>
                  <div className="prediction-value">
                    {result.predictions.occupancy.percentage.toFixed(1)}%
                  </div>
                  <div className="prediction-label">Occupancy Rate</div>
                  <div className="prediction-sub-value">
                    ~{formatNumber(result.predictions.rooms.value)} room nights
                  </div>
                </>
              ) : (
                <>
                  <div className="prediction-value">
                    {formatNumber(result.predictions.rooms.value)}
                  </div>
                  <div className="prediction-label">Expected Room Nights</div>
                </>
              )}
            </div>
            <div className="card-footer">
              <div className={`confidence-badge confidence-${getConfidenceColor(result.predictions.rooms.confidence)}`}>
                <span className="confidence-icon">📊</span>
                <span>{result.predictions.rooms.confidence.toFixed(1)}% Confidence</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="metadata-card">
        <h3 className="metadata-title">
          <span className="metadata-icon">⚙️</span>
          Prediction Metadata
        </h3>
        <div className="metadata-grid">
          <div className="metadata-item">
            <span className="metadata-label">Model Version</span>
            <span className="metadata-value">{result.metadata.modelVersion}</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Model Type</span>
            <span className="metadata-value">
              {result.metadata.modelType === 'xgb' ? 'XGBoost' : 'Random Forest'}
            </span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Processing Time</span>
            <span className="metadata-value">{result.metadata.processingTime}ms</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Overall Accuracy</span>
            <span className="metadata-value">{(result.metadata.accuracy * 100).toFixed(0)}%</span>
          </div>
          <div className="metadata-item">
            <span className="metadata-label">Prediction ID</span>
            <span className="metadata-value code">{result._id}</span>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="result-actions">
        <button
          onClick={onNewPrediction}
          className="action-button primary-action"
        >
          <span className="button-icon">🎯</span>
          <span>Create New Prediction</span>
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

export default PredictionResult