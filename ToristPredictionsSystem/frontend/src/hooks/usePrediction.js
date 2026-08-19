import { useState, useCallback } from 'react'
import * as predictionService from '../services/prediction.service'

export const usePrediction = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const createPrediction = useCallback(async (data) => {
    setLoading(true)
    setError(null)
    try {
      const result = await predictionService.createPrediction(data)
      return result
    } catch (err) {
      setError(err.response?.data?.message || 'Prediction failed')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  const getHistory = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await predictionService.getPredictionHistory()
      return result
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to fetch history')
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  return {
    createPrediction,
    getHistory,
    loading,
    error
  }
}