const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { protect, authorize } = require('../middleware/auth.middleware');
const {
  uploadFile,
  getAllFiles,
  getAvailableFiles,
  downloadFile,
  deleteFile,
  toggleFileStatus,
  getAllUsers,
  getUser,
  createUser,
  updateUser,
  deleteUser,
  updateUserRole,
  getAdminStats,
  uploadCSVForPrediction,
  getAllPredictions,
  deletePrediction
} = require('../controllers/admin.controller');

const router = express.Router();

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, '../../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    // Create unique filename with timestamp
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'csv-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const fileFilter = (req, file, cb) => {
  // Accept CSV files only
  if (file.mimetype === 'text/csv' || 
      file.mimetype === 'application/vnd.ms-excel' ||
      file.originalname.endsWith('.csv')) {
    cb(null, true);
  } else {
    cb(new Error('Only CSV files are allowed!'), false);
  }
};

const upload = multer({
  storage: storage,
  fileFilter: fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
});

// Public route for users to get available files (requires authentication)
router.get('/files/available', protect, getAvailableFiles);

// Public route for users to download files (requires authentication)
router.get('/files/download/:id', protect, downloadFile);

// Admin only routes
router.use(protect, authorize('admin'));

// File management routes
router.post('/upload', upload.single('file'), uploadFile);
router.get('/files', getAllFiles);
router.delete('/files/:id', deleteFile);
router.patch('/files/:id/toggle', toggleFileStatus);

// User management routes
router.get('/users', getAllUsers);
router.post('/users', createUser);
router.get('/users/:id', getUser);
router.put('/users/:id', updateUser);
router.delete('/users/:id', deleteUser);
router.patch('/users/:id/role', updateUserRole);

// CSV prediction route
router.post('/csv-predict', upload.single('file'), uploadCSVForPrediction);

// Prediction management routes
router.get('/predictions', getAllPredictions);
router.delete('/predictions/:id', deletePrediction);

// Dashboard stats
router.get('/stats', getAdminStats);

module.exports = router;
