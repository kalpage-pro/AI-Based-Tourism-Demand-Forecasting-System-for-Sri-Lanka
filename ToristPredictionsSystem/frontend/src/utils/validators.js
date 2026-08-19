export const validateEmail = (email) => {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return re.test(email)
}

export const validatePassword = (password) => {
  // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
  return password.length >= 8
}

export const validatePredictionForm = (formData) => {
  const errors = {}

  if (!formData.destination) {
    errors.destination = 'Destination is required'
  }

  if (!formData.month) {
    errors.month = 'Month is required'
  }

  if (!formData.year || formData.year < 2000 || formData.year > 2100) {
    errors.year = 'Please enter a valid year'
  }

  if (!formData.avgTemperature) {
    errors.avgTemperature = 'Temperature is required'
  }

  if (!formData.rainfall || formData.rainfall < 0) {
    errors.rainfall = 'Rainfall must be a positive number'
  }

  if (!formData.currency || formData.currency <= 0) {
    errors.currency = 'Currency rate must be greater than 0'
  }

  if (!formData.attractions || formData.attractions < 0) {
    errors.attractions = 'Attractions must be a positive number'
  }

  if (!formData.events || formData.events < 0) {
    errors.events = 'Events must be a positive number'
  }

  return {
    isValid: Object.keys(errors).length === 0,
    errors
  }
}