const express = require('express');
const cors = require('cors');
const path = require('path');
const errorMiddleware = require('./middleware/error.middleware');

// Route imports
const authRoutes = require('./routes/auth.routes');
const predictionRoutes = require('./routes/prediction.routes');
const dashboardRoutes = require('./routes/dashboard.routes');
const healthRoutes = require('./routes/health.routes');
const adminRoutes = require('./routes/admin.routes');
const analyticsRoutes = require('./routes/analytics.routes');
const scenarioRoutes = require('./routes/scenario.routes');
const exportRoutes = require('./routes/export.routes');
const destinationRoutes = require('./routes/destination.routes');

const app = express();

// CORS Configuration - Allow frontend to access backend
const corsOptions = {
  origin: ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:3000', 'http://127.0.0.1:5173'],
  credentials: true,
  optionsSuccessStatus: 200,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
};

// Middleware
app.use(cors(corsOptions));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Add request logging for debugging
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`);
  next();
});

// Routes
app.use('/api/v1/auth', authRoutes);
app.use('/api/v1/predictions', predictionRoutes);
app.use('/api/v1/dashboard', dashboardRoutes);
app.use('/api/v1/health', healthRoutes);
app.use('/api/v1/admin', adminRoutes);
app.use('/api/v1/analytics', analyticsRoutes);
app.use('/api/v1/scenarios', scenarioRoutes);
app.use('/api/v1/export', exportRoutes);
app.use('/api/v1/destinations', destinationRoutes);

// Serve uploaded files statically (for authorized downloads)
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// Welcome route
app.get('/', (req, res) => {
  res.json({
    message: '🇱🇰 Welcome to Sri Lanka Tourist Prediction System API',
    version: '2.0.0',
    endpoints: {
      auth: '/api/v1/auth',
      predictions: '/api/v1/predictions',
      dashboard: '/api/v1/dashboard',
      health: '/api/v1/health',
      admin: '/api/v1/admin',
      analytics: '/api/v1/analytics',
      scenarios: '/api/v1/scenarios',
      export: '/api/v1/export',
      destinations: '/api/v1/destinations'
    }
  });
});

// Error handling middleware (should be last)
app.use(errorMiddleware);

module.exports = app;