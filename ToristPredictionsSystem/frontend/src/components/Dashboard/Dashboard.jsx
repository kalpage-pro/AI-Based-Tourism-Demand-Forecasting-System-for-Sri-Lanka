import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import StatsCard from './StatsCard'
import TrendChart from './TrendChart'
import Card from '../Common/Card'
import Button from '../Common/Button'
import Loading from '../Common/Loading'
import dashboardService from '../../services/dashboard.service'
import predictionService from '../../services/prediction.service'
import { formatNumber, formatCurrency } from '../../utils/helpers'
import './Dashboard.css'

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [recentPredictions, setRecentPredictions] = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)
      const predictions = await predictionService.getPredictionHistory()
      
      // Calculate stats from recent predictions
      if (predictions && predictions.length > 0) {
        const latest = predictions.slice(0, 5)
        setRecentPredictions(latest)
        
        // Calculate aggregate stats
        const totalArrivals = predictions.reduce((sum, p) => 
          sum + (p.predictions.touristArrivals?.value || 0), 0)
        const totalRevenue = predictions.reduce((sum, p) => 
          sum + (p.predictions.revenue?.value || 0), 0)
        const avgAccuracy = predictions.reduce((sum, p) => 
          sum + (p.metadata?.accuracy || 0), 0) / predictions.length
        
        setStats({
          totalPredictions: predictions.length,
          totalArrivals,
          totalRevenue,
          avgAccuracy: avgAccuracy * 100,
          recentCount: latest.length
        })
      }
    } catch (error) {
      console.error('Dashboard load error:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <Loading message="Loading dashboard..." />
  }

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <div>
          <h1 className="dashboard-title">🇱🇰 Sri Lanka Tourism Dashboard</h1>
          <p className="dashboard-subtitle">Tourist prediction analytics and insights</p>
        </div>
        <Button onClick={() => navigate('/predict')} icon="🔮">
          New Prediction
        </Button>
      </div>

      <div className="stats-grid">
        <StatsCard
          title="Total Predictions"
          value={stats?.totalPredictions || 0}
          icon="📊"
          color="blue"
          trend="+12%"
        />
        <StatsCard
          title="Predicted Arrivals"
          value={formatNumber(stats?.totalArrivals || 0)}
          icon="✈️"
          color="green"
          trend="+8%"
        />
        <StatsCard
          title="Predicted Revenue"
          value={formatCurrency(stats?.totalRevenue || 0)}
          icon="💰"
          color="purple"
          trend="+15%"
        />
        <StatsCard
          title="Model Accuracy"
          value={`${stats?.avgAccuracy?.toFixed(1) || 0}%`}
          icon="🎯"
          color="orange"
          trend="stable"
        />
      </div>

      <div className="dashboard-content">
        <div className="dashboard-main">
          <Card title="Recent Predictions" icon="📈">
            {recentPredictions.length > 0 ? (
              <div className="predictions-list">
                {recentPredictions.map((prediction) => (
                  <div key={prediction._id} className="prediction-item">
                    <div className="prediction-date">
                      <span className="month">
                        {getMonthName(prediction.inputData.month)} {prediction.inputData.year}
                      </span>
                      <span className="type-badge">{prediction.predictionType}</span>
                    </div>
                    <div className="prediction-values">
                      {prediction.predictions.touristArrivals?.value > 0 && (
                        <div className="value-item">
                          <span className="label">Arrivals:</span>
                          <span className="value">
                            {formatNumber(prediction.predictions.touristArrivals.value)}
                          </span>
                        </div>
                      )}
                      {prediction.predictions.revenue?.value > 0 && (
                        <div className="value-item">
                          <span className="label">Revenue:</span>
                          <span className="value">
                            {formatCurrency(prediction.predictions.revenue.value)}
                          </span>
                        </div>
                      )}
                      {prediction.predictions.rooms?.value > 0 && (
                        <div className="value-item">
                          <span className="label">Rooms:</span>
                          <span className="value">
                            {formatNumber(prediction.predictions.rooms.value)}
                          </span>
                        </div>
                      )}
                    </div>
                    <div className="prediction-meta">
                      <span className="accuracy">
                        {(prediction.metadata.accuracy * 100).toFixed(0)}% accuracy
                      </span>
                      <span className="time">
                        {new Date(prediction.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <p>No predictions yet. Create your first prediction!</p>
                <Button onClick={() => navigate('/predict')}>
                  Create Prediction
                </Button>
              </div>
            )}
          </Card>
        </div>

        <div className="dashboard-sidebar">
          <Card title="Quick Actions" icon="⚡">
            <div className="quick-actions">
              <button 
                className="action-button"
                onClick={() => navigate('/predict')}
              >
                <span className="action-icon">🔮</span>
                <span className="action-text">New Prediction</span>
              </button>
              <button 
                className="action-button"
                onClick={() => navigate('/history')}
              >
                <span className="action-icon">📜</span>
                <span className="action-text">View History</span>
              </button>
            </div>
          </Card>

          <Card title="Model Info" icon="🤖" glow>
            <div className="model-info">
              <div className="info-row">
                <span className="info-label">Version:</span>
                <span className="info-value">1.0.0</span>
              </div>
              <div className="info-row">
                <span className="info-label">Algorithm:</span>
                <span className="info-value">Random Forest</span>
              </div>
              <div className="info-row">
                <span className="info-label">Avg Processing:</span>
                <span className="info-value">2.5s</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function getMonthName(month) {
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return months[month - 1] || 'Unknown'
}

export default Dashboard