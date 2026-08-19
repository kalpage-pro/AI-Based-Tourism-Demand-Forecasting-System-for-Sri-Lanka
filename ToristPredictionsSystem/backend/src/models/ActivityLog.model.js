const mongoose = require('mongoose');

const activityLogSchema = new mongoose.Schema({
  user: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  action: {
    type: String,
    required: true,
    enum: [
      'LOGIN',
      'LOGOUT', 
      'REGISTER',
      'PREDICTION_CREATED',
      'BATCH_PREDICTION_CREATED',
      'FILE_UPLOADED',
      'FILE_DELETED',
      'FILE_DOWNLOADED',
      'MODEL_RETRAINED',
      'USER_ROLE_CHANGED',
      'PROFILE_UPDATED',
      'EXPORT_GENERATED',
      'SCENARIO_SIMULATION'
    ]
  },
  details: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  ipAddress: {
    type: String
  },
  userAgent: {
    type: String
  },
  status: {
    type: String,
    enum: ['SUCCESS', 'FAILED'],
    default: 'SUCCESS'
  }
}, {
  timestamps: true
});

// Index for efficient queries
activityLogSchema.index({ user: 1, createdAt: -1 });
activityLogSchema.index({ action: 1, createdAt: -1 });
activityLogSchema.index({ createdAt: -1 });

// Static method to log activity
activityLogSchema.statics.log = async function(userId, action, details = {}, req = null) {
  try {
    const logEntry = {
      user: userId,
      action,
      details,
      status: 'SUCCESS'
    };
    
    if (req) {
      logEntry.ipAddress = req.ip || req.connection?.remoteAddress;
      logEntry.userAgent = req.get('User-Agent');
    }
    
    return await this.create(logEntry);
  } catch (error) {
    console.error('Failed to log activity:', error);
    return null;
  }
};

module.exports = mongoose.model('ActivityLog', activityLogSchema);
