# 🇱🇰 AI-Based Tourism Demand Forecasting System for Sri Lanka

A comprehensive full-stack application for predicting tourist arrivals, tourism revenue, and hotel occupancy rates using machine learning models.

## 📖 Project Overview

This system helps tourism stakeholders in Sri Lanka forecast tourism demand using AI/ML models trained on historical data. It features:

- **Multiple ML Models**: Random Forest and XGBoost for each prediction target
- **Interactive Analytics**: Feature importance, model comparison, seasonal patterns
- **Scenario Simulation**: What-if analysis for tourism planning
- **Export Functionality**: PDF reports and CSV exports
- **Real-time Predictions**: API-based predictions with confidence scores

## 🏗️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 18, Vite, Recharts, CSS3 |
| **Backend** | Node.js, Express, MongoDB, JWT Auth |
| **ML Service** | Python 3, scikit-learn, XGBoost, pandas |
| **Database** | MongoDB with Mongoose ODM |

## 📁 Project Structure

```
ToristPredictionsSystem/
├── backend/                 # Node.js Express API
│   ├── src/
│   │   ├── controllers/     # API controllers
│   │   ├── models/          # MongoDB schemas
│   │   ├── routes/          # API routes
│   │   ├── services/        # Business logic
│   │   └── seeds/           # Database seeders
│   └── package.json
├── frontend/                # React Vite application
│   ├── src/
│   │   ├── components/      # Reusable components
│   │   ├── pages/           # Page components
│   │   ├── services/        # API services
│   │   ├── hooks/           # Custom React hooks
│   │   └── context/         # React context providers
│   └── package.json
├── Ml/                      # Machine Learning service
│   ├── T/                   # Trained models + scripts
│   │   ├── *.pkl            # Trained model files
│   │   ├── predict.py       # Prediction script
│   │   ├── feature_importance.py
│   │   └── evaluate_models.py
│   └── src/                 # Training scripts
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- Python 3.9+
- MongoDB 6.0+

### 1. Clone & Setup

```bash
# Backend setup
cd backend
npm install
cp .env.example .env
# Edit .env with your MongoDB URI and JWT secret

# Frontend setup
cd ../frontend
npm install

# Python ML setup
cd ../Ml
pip install -r requirements.txt
```

### 2. Seed Database

```bash
cd backend
node src/seeds/admin.seed.js        # Create admin user
node src/seeds/historicalData.seed.js  # Load historical data
```

### 3. Start Services

```bash
# Terminal 1 - Backend (http://localhost:5000)
cd backend
npm run dev

# Terminal 2 - Frontend (http://localhost:5173)
cd frontend
npm run dev
```

### 4. Default Login

- **Admin**: admin@tourist.lk / admin123
- **User**: Register a new account

## 🔌 API Endpoints

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Register new user |
| POST | `/api/v1/auth/login` | User login |
| GET | `/api/v1/auth/me` | Get current user |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predictions` | Create single prediction |
| POST | `/api/v1/predictions/batch` | Create batch predictions |
| GET | `/api/v1/predictions` | Get user's predictions |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/feature-importance` | Get feature importance |
| GET | `/api/v1/analytics/models/compare` | Compare RF vs XGBoost |
| GET | `/api/v1/analytics/seasonal-patterns` | Get seasonal patterns |
| GET | `/api/v1/analytics/yearly-trends` | Get yearly trends |
| GET | `/api/v1/analytics/forecast` | Get 12-month forecast |

### Scenarios
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/scenarios/simulate` | Run scenario simulation |
| GET | `/api/v1/scenarios/templates` | Get scenario templates |
| POST | `/api/v1/scenarios/what-if` | Run what-if analysis |

### Export
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/export/predictions/csv` | Export to CSV |
| GET | `/api/v1/export/predictions/pdf` | Export to PDF report |
| GET | `/api/v1/export/historical` | Export historical data |

## 🤖 ML Models

### Model Types
- **Random Forest (rf)**: Best for interpretability
- **XGBoost (xgb)**: Best for accuracy

### Prediction Targets
- **Tourist Arrivals**: Monthly visitor count
- **Tourism Revenue**: Monthly revenue in USD
- **Hotel Occupancy**: Occupancy rate (0-1)

### Key Features (22 total)
| Category | Features |
|----------|----------|
| Temporal | year, month, quarter, month_sin, month_cos |
| Economic | dollarrate, cpi, airfare_index |
| Weather | apparent_temperature, sunshine, rain, precipitation_hours |
| Infrastructure | num_establishments, num_rooms |
| Lag Features | arrivals_lag1-12, revenue_lag1, rolling averages |

## 📊 Frontend Pages

| Page | Description |
|------|-------------|
| Dashboard | Overview with stats and trends |
| Predictions | Single and batch prediction forms |
| Analytics | Feature importance, model comparison, charts |
| Scenarios | What-if analysis and scenario simulation |
| History | View past predictions |
| Reports | Export data to CSV/PDF |
| Admin | User management, file uploads, system stats |

## 🔧 Environment Variables

### Backend (.env)
```env
NODE_ENV=development
PORT=5000
MONGODB_URI=mongodb://localhost:27017/tourist_prediction
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRE=7d
PYTHON_PATH=python
```

## 🧪 Testing Predictions

```bash
# CLI test
cd Ml/T
python predict.py --year 2025 --month 6 --model rf --type arrivals

# With custom parameters
python predict.py --year 2025 --month 12 --model xgb --type all \
  --dollar-rate 300 --apparent-temperature 28
```

## 📈 Model Performance

Models are trained on Sri Lanka tourism data (2018-2024) with:
- Cross-validation for generalization
- Hyperparameter tuning via GridSearchCV
- Feature engineering for time series patterns

## 🎓 For Viva Presentation

1. **Login** → Show auth flow
2. **Make Prediction** → Explain ML models
3. **Analytics Page** → Show feature importance
4. **Scenario Simulation** → What-if analysis
5. **Export Report** → Generate PDF
6. **Admin Panel** → System management
7. **Architecture Diagram** → Explain tech stack

## 📝 License

MIT License - Built for Final Year Project
