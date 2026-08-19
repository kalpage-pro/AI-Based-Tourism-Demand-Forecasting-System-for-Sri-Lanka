import { useState, useEffect } from 'react'
import { getPredictionHistory } from '../../services/prediction.service'
import Card from '../Common/Card'
import Loading from '../Common/Loading'
import './Prediction.css'

const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function PredictionHistory() {
  const [predictions, setPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      setError(null)
      const data = await getPredictionHistory()
      console.log('Fetched predictions:', data)
      setPredictions(Array.isArray(data) ? data : [])
    } catch (error) {
      console.error('Failed to fetch history:', error)
      setError('Failed to load prediction history')
      setPredictions([])
    } finally {
      setLoading(false)
    }
  }

  const filteredPredictions = predictions.filter(p => {
    if (filter === 'all') return true
    if (filter === 'recent') {
      const weekAgo = new Date()
      weekAgo.setDate(weekAgo.getDate() - 7)
      return new Date(p.createdAt) > weekAgo
    }
    return true
  })

  const getPredictionLabel = (type) => {
    const labels = {
      'tourist_arrivals': '👥 Tourist Arrivals',
      'revenue': '💰 Revenue',
      'rooms': '🏨 Room Occupancy',
      'all': '📊 All Metrics'
    }
    return labels[type] || type
  }

  const formatPredictionValue = (prediction) => {
    if (!prediction.predictions) return 'N/A'
    
    const { touristArrivals, revenue, rooms } = prediction.predictions
    
    if (prediction.predictionType === 'tourist_arrivals' && touristArrivals) {
      return `${Math.round(touristArrivals.value || 0).toLocaleString()} tourists`
    }
    if (prediction.predictionType === 'revenue' && revenue) {
      return `LKR ${Math.round(revenue.value || 0).toLocaleString()}`
    }
    if (prediction.predictionType === 'rooms' && rooms) {
      return `${Math.round(rooms.value || 0).toLocaleString()} rooms`
    }
    if (prediction.predictionType === 'all' && touristArrivals) {
      return `${Math.round(touristArrivals.value || 0).toLocaleString()} tourists`
    }
    
    return 'N/A'
  }

  const getConfidence = (prediction) => {
    if (!prediction.predictions) return null
    
    const { touristArrivals, revenue, rooms } = prediction.predictions
    
    if (prediction.predictionType === 'tourist_arrivals' && touristArrivals) {
      return touristArrivals.confidence
    }
    if (prediction.predictionType === 'revenue' && revenue) {
      return revenue.confidence
    }
    if (prediction.predictionType === 'rooms' && rooms) {
      return rooms.confidence
    }
    if (prediction.predictionType === 'all' && touristArrivals) {
      return touristArrivals.confidence
    }
    
    return null
  }

  if (loading) return <Loading message="Loading history..." />

  return (
    <Card 
      title="Prediction History" 
      icon="📜"
      actions={
        <select 
          className="filter-select"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        >
          <option value="all">All Time</option>
          <option value="recent">Last 7 Days</option>
        </select>
      }
    >
      {error && <div className="error-message">{error}</div>}
      
      <div className="history-list">
        {filteredPredictions.length > 0 ? (
          filteredPredictions.map((prediction) => (
            <div key={prediction._id} className="history-item">
              <div className="history-header">
                <div className="history-destination">
                  <span className="history-icon">📅</span>
                  <span>
                    {monthNames[prediction.inputData?.month - 1] || 'N/A'} {prediction.inputData?.year || 'N/A'}
                  </span>
                </div>
                <span className="history-badge">
                  {getPredictionLabel(prediction.predictionType)}
                </span>
              </div>
              <div className="history-body">
                <div className="history-stat">
                  <span className="stat-label">Predicted</span>
                  <span className="stat-value">
                    {formatPredictionValue(prediction)}
                  </span>
                </div>
                {getConfidence(prediction) && (
                  <div className="history-stat">
                    <span className="stat-label">Confidence</span>
                    <span className="stat-value">{Math.round(getConfidence(prediction))}%</span>
                  </div>
                )}
                <div className="history-stat">
                  <span className="stat-label">Date</span>
                  <span className="stat-value">
                    {new Date(prediction.createdAt).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="empty-state">
            <span className="empty-icon">📭</span>
            <p>No predictions found</p>
            <p className="empty-hint">Make your first prediction to see it here!</p>
          </div>
        )}
      </div>
    </Card>
  )
}

export default PredictionHistory