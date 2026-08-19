import PredictionHistory from '../components/Prediction/PredictionHistory'
import './Pages.css'

function HistoryPage() {
  return (
    <div className="page history-page">
      <div className="page-header">
        <h1 className="page-title">Prediction History</h1>
        <p className="page-subtitle">View all your past predictions</p>
      </div>

      <div className="history-content">
        <PredictionHistory />
      </div>
    </div>
  )
}

export default HistoryPage