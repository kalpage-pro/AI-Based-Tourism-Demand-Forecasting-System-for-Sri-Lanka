/**
 * Admin Seeder Script
 * 
 * This script creates an initial admin user for the Tourist Prediction System.
 * Run this script once to set up the admin account.
 * 
 * Usage: node src/seeds/admin.seed.js
 */

require('dotenv').config();
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

// Admin credentials - CHANGE THESE IN PRODUCTION!
const ADMIN_EMAIL = 'admin@touristprediction.lk';
const ADMIN_PASSWORD = 'Admin@123';
const ADMIN_NAME = 'System Administrator';

const userSchema = new mongoose.Schema({
  name: String,
  email: String,
  password: String,
  role: { type: String, default: 'user' },
  organization: String,
  createdAt: { type: Date, default: Date.now }
});

const User = mongoose.model('User', userSchema);

async function seedAdmin() {
  try {
    // Connect to MongoDB
    const mongoUri = process.env.MONGODB_URI || process.env.MONGO_URI || 'mongodb://localhost:27017/tourist_prediction';
    await mongoose.connect(mongoUri);
    console.log('📦 Connected to MongoDB');

    // Check if admin already exists
    const existingAdmin = await User.findOne({ email: ADMIN_EMAIL });
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(ADMIN_PASSWORD, salt);
    
    if (existingAdmin) {
      console.log('⚠️  Admin user already exists!');
      console.log(`   Email: ${ADMIN_EMAIL}`);
      
      // Update role and password
      existingAdmin.role = 'admin';
      existingAdmin.password = hashedPassword;
      await existingAdmin.save();
      console.log('✅ Updated admin role and password');
    } else {

      // Create admin user
      const admin = await User.create({
        name: ADMIN_NAME,
        email: ADMIN_EMAIL,
        password: hashedPassword,
        role: 'admin',
        organization: 'Sri Lanka Tourism Development Authority'
      });

      console.log('✅ Admin user created successfully!');
      console.log('');
      console.log('📋 Admin Credentials:');
      console.log(`   Email: ${ADMIN_EMAIL}`);
      console.log(`   Password: ${ADMIN_PASSWORD}`);
      console.log('');
      console.log('⚠️  IMPORTANT: Change these credentials in production!');
    }

    await mongoose.disconnect();
    console.log('📦 Disconnected from MongoDB');
    process.exit(0);
  } catch (error) {
    console.error('❌ Error seeding admin:', error.message);
    process.exit(1);
  }
}

seedAdmin();
