import { useState } from 'react'
import PredictionService from '../services/prediction.service'
import PredictionForm from '../components/Prediction/PredictionForm'
import PredictionResult from '../components/Prediction/PredictionResult'
import BatchPredictionForm from '../components/Prediction/BatchPredictionForm'
import BatchPredictionResults from '../components/Prediction/BatchPredictionResults'
import './PredictionPage.css'

function PredictionPage() {
  const [activeTab, setActiveTab] = useState('single') // 'single' or 'batch'
  const [result, setResult] = useState(null)
  const [batchResults, setBatchResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSinglePrediction = async (formData) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await PredictionService.createPrediction(formData)
      
      if (response.success) {
        setResult(response.data)
      } else {
        setError('Prediction failed. Please try again.')
      }
    } catch (err) {
      console.error('Prediction error:', err)
      setError(err.response?.data?.message || 'Failed to create prediction')
    } finally {
      setLoading(false)
    }
  }

  const handleBatchPrediction = async (formData) => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await PredictionService.batchPredictions(formData)
      
      if (response.success) {
        setBatchResults(response.data)
      } else {
        setError('Batch prediction failed. Please try again.')
      }
    } catch (err) {
      console.error('Batch prediction error:', err)
      setError(err.response?.data?.message || 'Failed to create batch predictions')
    } finally {
      setLoading(false)
    }
  }

  const handleNewPrediction = () => {
    setResult(null)
    setBatchResults(null)
    setError(null)
  }

  const handleTabChange = (tab) => {
    setActiveTab(tab)
    handleNewPrediction()
  }

  return (
    <div className="prediction-page">
      {/* Animated Background */}
      <div className="prediction-background">
        <div className="prediction-orb orb-1"></div>
        <div className="prediction-orb orb-2"></div>
        <div className="prediction-orb orb-3"></div>
      </div>

      <div className="prediction-container">
        {/* Header */}
        <header className="prediction-header">
          <div className="header-badge">
            <span className="badge-pulse"></span>
            <span>AI-Powered Predictions</span>
          </div>
          <h1 className="prediction-title">
            <span className="title-icon">🎯</span>
            <span className="title-gradient">Tourist</span> Predictions
          </h1>
          <p className="prediction-subtitle">
            Create accurate predictions using advanced machine learning algorithms
          </p>
        </header>

        {/* Tab Navigation */}
        <div className="prediction-tabs">
          <button
            className={`tab-button ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => handleTabChange('single')}
          >
            <span className="tab-icon">📊</span>
            <span>Single Prediction</span>
          </button>
          <button
            className={`tab-button ${activeTab === 'batch' ? 'active' : ''}`}
            onClick={() => handleTabChange('batch')}
          >
            <span className="tab-icon">📈</span>
            <span>Batch Predictions</span>
          </button>
          <div className={`tab-indicator ${activeTab}`}></div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="error-alert">
            <div className="alert-icon">⚠️</div>
            <div className="alert-content">
              <h4 className="alert-title">Error</h4>
              <p className="alert-message">{error}</p>
            </div>
            <button className="alert-close" onClick={() => setError(null)}>
              ✕
            </button>
          </div>
        )}

        {/* Content Area */}
        <div className="prediction-content">
          {activeTab === 'single' ? (
            <>
              {!result ? (
                <PredictionForm 
                  onSubmit={handleSinglePrediction} 
                  loading={loading} 
                />
              ) : (
                <PredictionResult 
                  result={result} 
                  onNewPrediction={handleNewPrediction} 
                />
              )}
            </>
          ) : (
            <>
              {!batchResults ? (
                <BatchPredictionForm 
                  onSubmit={handleBatchPrediction} 
                  loading={loading} 
                />
              ) : (
                <BatchPredictionResults 
                  results={batchResults} 
                  onNewPrediction={handleNewPrediction} 
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default PredictionPage