export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api'

export const ROUTES = {
  HOME: '/',
  LOGIN: '/login',
  REGISTER: '/register',
  PREDICT: '/predict',
  HISTORY: '/history',
  PROFILE: '/profile'
}

export const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

export const CHART_COLORS = {
  primary: '#FFD700',
  secondary: '#FFC107',
  accent: '#FFEB3B',
  success: '#10b981',
  error: '#ef4444',
  warning: '#f59e0b'
}