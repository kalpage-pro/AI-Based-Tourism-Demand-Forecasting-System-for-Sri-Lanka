const File = require('../models/File.model');
const User = require('../models/User.model');
const Prediction = require('../models/Prediction.model');
const path = require('path');
const fs = require('fs');
const bcrypt = require('bcryptjs');
const csv = require('csv-parser');

// @desc    Upload CSV file (Admin only)
// @route   POST /api/v1/admin/upload
// @access  Private/Admin
exports.uploadFile = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'Please upload a CSV file'
      });
    }

    // Validate file type
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
    if (!allowedTypes.includes(req.file.mimetype) && !req.file.originalname.endsWith('.csv')) {
      // Remove uploaded file
      fs.unlinkSync(req.file.path);
      return res.status(400).json({
        success: false,
        message: 'Only CSV files are allowed'
      });
    }

    const file = await File.create({
      filename: req.file.filename,
      originalName: req.file.originalname,
      description: req.body.description || '',
      filePath: req.file.path,
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      uploadedBy: req.user.id
    });

    res.status(201).json({
      success: true,
      message: 'File uploaded successfully',
      data: file
    });
  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get all uploaded files (Admin only)
// @route   GET /api/v1/admin/files
// @access  Private/Admin
exports.getAllFiles = async (req, res) => {
  try {
    const files = await File.find()
      .populate('uploadedBy', 'name email')
      .sort({ createdAt: -1 });

    res.status(200).json({
      success: true,
      count: files.length,
      data: files
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get active files for users
// @route   GET /api/v1/admin/files/available
// @access  Private
exports.getAvailableFiles = async (req, res) => {
  try {
    const files = await File.find({ isActive: true })
      .select('filename originalName description fileSize createdAt downloadCount')
      .sort({ createdAt: -1 });

    res.status(200).json({
      success: true,
      count: files.length,
      data: files
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Download file
// @route   GET /api/v1/admin/files/download/:id
// @access  Private
exports.downloadFile = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);

    if (!file) {
      return res.status(404).json({
        success: false,
        message: 'File not found'
      });
    }

    if (!file.isActive) {
      return res.status(403).json({
        success: false,
        message: 'This file is no longer available for download'
      });
    }

    // Check if file exists on disk
    if (!fs.existsSync(file.filePath)) {
      return res.status(404).json({
        success: false,
        message: 'File not found on server'
      });
    }

    // Increment download count
    file.downloadCount += 1;
    await file.save();

    // Send file for download
    res.download(file.filePath, file.originalName);
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Delete file (Admin only)
// @route   DELETE /api/v1/admin/files/:id
// @access  Private/Admin
exports.deleteFile = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);

    if (!file) {
      return res.status(404).json({
        success: false,
        message: 'File not found'
      });
    }

    // Delete file from disk
    if (fs.existsSync(file.filePath)) {
      fs.unlinkSync(file.filePath);
    }

    await File.findByIdAndDelete(req.params.id);

    res.status(200).json({
      success: true,
      message: 'File deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Toggle file active status (Admin only)
// @route   PATCH /api/v1/admin/files/:id/toggle
// @access  Private/Admin
exports.toggleFileStatus = async (req, res) => {
  try {
    const file = await File.findById(req.params.id);

    if (!file) {
      return res.status(404).json({
        success: false,
        message: 'File not found'
      });
    }

    file.isActive = !file.isActive;
    await file.save();

    res.status(200).json({
      success: true,
      message: `File ${file.isActive ? 'activated' : 'deactivated'} successfully`,
      data: file
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get all users (Admin only)
// @route   GET /api/v1/admin/users
// @access  Private/Admin
exports.getAllUsers = async (req, res) => {
  try {
    const users = await User.find()
      .select('-password')
      .sort({ createdAt: -1 });

    res.status(200).json({
      success: true,
      count: users.length,
      data: users
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Update user role (Admin only)
// @route   PATCH /api/v1/admin/users/:id/role
// @access  Private/Admin
exports.updateUserRole = async (req, res) => {
  try {
    const { role } = req.body;

    if (!['user', 'admin'].includes(role)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid role. Must be either "user" or "admin"'
      });
    }

    const user = await User.findByIdAndUpdate(
      req.params.id,
      { role },
      { new: true, runValidators: true }
    ).select('-password');

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    res.status(200).json({
      success: true,
      message: `User role updated to ${role}`,
      data: user
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get admin dashboard stats
// @route   GET /api/v1/admin/stats
// @access  Private/Admin
exports.getAdminStats = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments();
    const totalAdmins = await User.countDocuments({ role: 'admin' });
    const totalFiles = await File.countDocuments();
    const activeFiles = await File.countDocuments({ isActive: true });
    const totalDownloads = await File.aggregate([
      { $group: { _id: null, total: { $sum: '$downloadCount' } } }
    ]);
    const totalPredictions = await Prediction.countDocuments();

    res.status(200).json({
      success: true,
      data: {
        totalUsers,
        totalAdmins,
        totalFiles,
        activeFiles,
        totalDownloads: totalDownloads[0]?.total || 0,
        totalPredictions
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Create user by admin
// @route   POST /api/v1/admin/users
// @access  Private/Admin
exports.createUser = async (req, res) => {
  try {
    const { name, email, password, role, organization } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({
        success: false,
        message: 'User with this email already exists'
      });
    }

    // Create user (password will be hashed by User model pre-save hook)
    const user = await User.create({
      name,
      email,
      password,
      role: role || 'user',
      organization: organization || 'Sri Lanka Tourism Development Authority'
    });

    res.status(201).json({
      success: true,
      message: 'User created successfully',
      data: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        organization: user.organization,
        createdAt: user.createdAt
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get single user
// @route   GET /api/v1/admin/users/:id
// @access  Private/Admin
exports.getUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id).select('-password');

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    res.status(200).json({
      success: true,
      data: user
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Update user by admin
// @route   PUT /api/v1/admin/users/:id
// @access  Private/Admin
exports.updateUser = async (req, res) => {
  try {
    const { name, email, role, organization, password } = req.body;

    const user = await User.findById(req.params.id);

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    // Update fields
    if (name) user.name = name;
    if (email) user.email = email;
    if (role) user.role = role;
    if (organization) user.organization = organization;
    if (password) user.password = password; // Will be hashed by pre-save hook

    await user.save();

    res.status(200).json({
      success: true,
      message: 'User updated successfully',
      data: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        organization: user.organization
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Delete user
// @route   DELETE /api/v1/admin/users/:id
// @access  Private/Admin
exports.deleteUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found'
      });
    }

    // Prevent deleting yourself
    if (user._id.toString() === req.user.id) {
      return res.status(400).json({
        success: false,
        message: 'You cannot delete your own account'
      });
    }

    await User.findByIdAndDelete(req.params.id);

    res.status(200).json({
      success: true,
      message: 'User deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Upload CSV for batch predictions (Admin only)
// @route   POST /api/v1/admin/csv-predict
// @access  Private/Admin
exports.uploadCSVForPrediction = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: 'Please upload a CSV file'
      });
    }

    const results = [];
    const errors = [];
    let rowNumber = 0;

    // Parse CSV file
    const parseCSV = () => {
      return new Promise((resolve, reject) => {
        fs.createReadStream(req.file.path)
          .pipe(csv())
          .on('data', (row) => {
            rowNumber++;
            try {
              // Extract data from CSV row
              const predictionData = {
                year: parseInt(row.year) || parseInt(row.Year),
                month: parseInt(row.month) || parseInt(row.Month),
                dollarRate: parseFloat(row.dollarRate || row.dollar_rate || row.DollarRate) || 320,
                predictionType: row.predictionType || row.prediction_type || 'all'
              };

              if (!predictionData.year || !predictionData.month) {
                errors.push({ row: rowNumber, error: 'Missing year or month' });
              } else {
                results.push(predictionData);
              }
            } catch (err) {
              errors.push({ row: rowNumber, error: err.message });
            }
          })
          .on('end', () => {
            resolve();
          })
          .on('error', (err) => {
            reject(err);
          });
      });
    };

    await parseCSV();

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    // Save the file record
    const file = await File.create({
      filename: req.file.filename,
      originalName: req.file.originalname,
      description: 'CSV for batch predictions',
      filePath: 'processed',
      fileSize: req.file.size,
      mimeType: req.file.mimetype,
      uploadedBy: req.user.id,
      isActive: false
    });

    res.status(200).json({
      success: true,
      message: `CSV parsed successfully. ${results.length} rows ready for prediction.`,
      data: {
        fileId: file._id,
        totalRows: rowNumber,
        validRows: results.length,
        errors: errors,
        predictionData: results
      }
    });
  } catch (error) {
    // Clean up file on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Get all predictions (Admin only)
// @route   GET /api/v1/admin/predictions
// @access  Private/Admin
exports.getAllPredictions = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    const predictions = await Prediction.find()
      .populate('user', 'name email')
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit);

    const total = await Prediction.countDocuments();

    res.status(200).json({
      success: true,
      count: predictions.length,
      total,
      page,
      pages: Math.ceil(total / limit),
      data: predictions
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};

// @desc    Delete prediction (Admin only)
// @route   DELETE /api/v1/admin/predictions/:id
// @access  Private/Admin
exports.deletePrediction = async (req, res) => {
  try {
    const prediction = await Prediction.findById(req.params.id);

    if (!prediction) {
      return res.status(404).json({
        success: false,
        message: 'Prediction not found'
      });
    }

    await Prediction.findByIdAndDelete(req.params.id);

    res.status(200).json({
      success: true,
      message: 'Prediction deleted successfully'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
};
