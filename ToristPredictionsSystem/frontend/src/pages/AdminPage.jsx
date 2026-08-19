import { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import adminService from '../services/admin.service';
import destinationService from '../services/destination.service';
import Loading from '../components/Common/Loading';
import Modal from '../components/Common/Modal';
import './AdminPage.css';

const CATEGORIES = ['beach', 'cultural', 'wildlife', 'adventure', 'hill-country', 'historical', 'religious', 'nature'];
const REGIONS = ['North', 'South', 'East', 'West', 'Central', 'North Central', 'North Western', 'Sabaragamuwa', 'Uva'];

function AdminPage() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState('files');
  const [stats, setStats] = useState(null);
  const [files, setFiles] = useState([]);
  const [users, setUsers] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [alert, setAlert] = useState(null);
  
  // Upload form state
  const [selectedFile, setSelectedFile] = useState(null);
  const [description, setDescription] = useState('');
  
  // CSV Prediction state
  const [csvFile, setCsvFile] = useState(null);
  const [csvResult, setCsvResult] = useState(null);
  const [uploadingCsv, setUploadingCsv] = useState(false);
  
  // User modal state
  const [showUserModal, setShowUserModal] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [userForm, setUserForm] = useState({
    name: '',
    email: '',
    password: '',
    role: 'user',
    organization: ''
  });

  // Destination state
  const [destinations, setDestinations] = useState([]);
  const [showDestModal, setShowDestModal] = useState(false);
  const [editingDest, setEditingDest] = useState(null);
  const [destForm, setDestForm] = useState({
    name: '',
    region: 'Central',
    description: '',
    category: 'cultural',
    highlights: '',
    bestTimeToVisit: '',
    popularity: 50,
    yearlyArrivals: 0,
    averageStayDays: 2,
    isFeatured: false
  });

  // Hotel modal state
  const [showHotelModal, setShowHotelModal] = useState(false);
  const [currentDestForHotel, setCurrentDestForHotel] = useState(null);
  const [hotelForm, setHotelForm] = useState({
    name: '',
    rating: 3,
    priceRange: 'mid-range',
    pricePerNight: 0,
    amenities: '',
    contact: ''
  });

  // Flight modal state
  const [showFlightModal, setShowFlightModal] = useState(false);
  const [currentDestForFlight, setCurrentDestForFlight] = useState(null);
  const [flightForm, setFlightForm] = useState({
    airline: '',
    from: 'Colombo',
    price: 0,
    duration: '',
    frequency: '',
    isEconomical: false
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsRes, filesRes, usersRes, destRes] = await Promise.all([
        adminService.getAdminStats(),
        adminService.getAllFiles(),
        adminService.getAllUsers(),
        destinationService.getAllDestinations({ limit: 100 }).catch(() => ({ data: [] }))
      ]);
      setStats(statsRes.data);
      setFiles(filesRes.data);
      setUsers(usersRes.data);
      setDestinations(destRes.data || []);
    } catch (error) {
      showAlert('error', 'Failed to load data');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPredictions = async () => {
    try {
      const predictionsRes = await adminService.getAllPredictions();
      setPredictions(predictionsRes.data);
    } catch (error) {
      console.error('Failed to load predictions:', error);
    }
  };

  const showAlert = (type, message) => {
    setAlert({ type, message });
    setTimeout(() => setAlert(null), 5000);
  };

  // User Management Functions
  const openCreateUserModal = () => {
    setEditingUser(null);
    setUserForm({
      name: '',
      email: '',
      password: '',
      role: 'user',
      organization: ''
    });
    setShowUserModal(true);
  };

  const openEditUserModal = (userToEdit) => {
    setEditingUser(userToEdit);
    setUserForm({
      name: userToEdit.name,
      email: userToEdit.email,
      password: '',
      role: userToEdit.role,
      organization: userToEdit.organization || ''
    });
    setShowUserModal(true);
  };

  const handleUserFormChange = (e) => {
    const { name, value } = e.target;
    setUserForm(prev => ({ ...prev, [name]: value }));
  };

  const handleUserSubmit = async (e) => {
    e.preventDefault();
    try {
      if (editingUser) {
        const updateData = { ...userForm };
        if (!updateData.password) delete updateData.password;
        await adminService.updateUser(editingUser._id, updateData);
        showAlert('success', 'User updated successfully');
      } else {
        await adminService.createUser(userForm);
        showAlert('success', 'User created successfully');
      }
      setShowUserModal(false);
      const usersRes = await adminService.getAllUsers();
      setUsers(usersRes.data);
      const statsRes = await adminService.getAdminStats();
      setStats(statsRes.data);
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Operation failed');
    }
  };

  const handleDeleteUser = async (userId) => {
    if (!confirm('Are you sure you want to delete this user?')) return;
    try {
      await adminService.deleteUser(userId);
      showAlert('success', 'User deleted successfully');
      const usersRes = await adminService.getAllUsers();
      setUsers(usersRes.data);
      const statsRes = await adminService.getAdminStats();
      setStats(statsRes.data);
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Failed to delete user');
    }
  };

  // CSV Prediction Functions
  const handleCsvFileSelect = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.csv')) {
      setCsvFile(file);
      setCsvResult(null);
    } else {
      showAlert('error', 'Please select a CSV file');
    }
  };

  const handleCsvUpload = async (e) => {
    e.preventDefault();
    if (!csvFile) {
      showAlert('error', 'Please select a CSV file');
      return;
    }
    try {
      setUploadingCsv(true);
      const formData = new FormData();
      formData.append('file', csvFile);
      const result = await adminService.uploadCSVForPrediction(formData);
      setCsvResult(result.data);
      showAlert('success', result.message);
      setCsvFile(null);
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'CSV upload failed');
    } finally {
      setUploadingCsv(false);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.endsWith('.csv')) {
        showAlert('error', 'Please select a CSV file');
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!selectedFile) {
      showAlert('error', 'Please select a file');
      return;
    }

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('description', description);

      await adminService.uploadFile(formData);
      showAlert('success', 'File uploaded successfully');
      setSelectedFile(null);
      setDescription('');
      
      // Refresh files list
      const filesRes = await adminService.getAllFiles();
      setFiles(filesRes.data);
      
      // Refresh stats
      const statsRes = await adminService.getAdminStats();
      setStats(statsRes.data);
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDownload = async (file) => {
    try {
      await adminService.downloadFile(file._id, file.originalName);
    } catch (error) {
      showAlert('error', 'Download failed');
    }
  };

  const handleToggleStatus = async (fileId) => {
    try {
      await adminService.toggleFileStatus(fileId);
      const filesRes = await adminService.getAllFiles();
      setFiles(filesRes.data);
      showAlert('success', 'File status updated');
    } catch (error) {
      showAlert('error', 'Failed to update status');
    }
  };

  const handleDeleteFile = async (fileId) => {
    if (!confirm('Are you sure you want to delete this file?')) return;
    
    try {
      await adminService.deleteFile(fileId);
      const filesRes = await adminService.getAllFiles();
      setFiles(filesRes.data);
      
      const statsRes = await adminService.getAdminStats();
      setStats(statsRes.data);
      
      showAlert('success', 'File deleted successfully');
    } catch (error) {
      showAlert('error', 'Failed to delete file');
    }
  };

  const handleRoleChange = async (userId, newRole) => {
    try {
      await adminService.updateUserRole(userId, newRole);
      const usersRes = await adminService.getAllUsers();
      setUsers(usersRes.data);
      
      const statsRes = await adminService.getAdminStats();
      setStats(statsRes.data);
      
      showAlert('success', 'User role updated');
    } catch (error) {
      showAlert('error', 'Failed to update user role');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  // Destination Management Functions
  const fetchDestinations = async () => {
    try {
      const destRes = await destinationService.getAllDestinations({ limit: 100 });
      setDestinations(destRes.data || []);
    } catch (error) {
      console.error('Failed to fetch destinations:', error);
    }
  };

  const openCreateDestModal = () => {
    setEditingDest(null);
    setDestForm({
      name: '',
      region: 'Central',
      description: '',
      category: 'cultural',
      highlights: '',
      bestTimeToVisit: '',
      popularity: 50,
      yearlyArrivals: 0,
      averageStayDays: 2,
      isFeatured: false
    });
    setShowDestModal(true);
  };

  const openEditDestModal = (dest) => {
    setEditingDest(dest);
    setDestForm({
      name: dest.name,
      region: dest.region,
      description: dest.description,
      category: dest.category,
      highlights: dest.highlights?.join(', ') || '',
      bestTimeToVisit: dest.bestTimeToVisit || '',
      popularity: dest.popularity || 50,
      yearlyArrivals: dest.yearlyArrivals || 0,
      averageStayDays: dest.averageStayDays || 2,
      isFeatured: dest.isFeatured || false
    });
    setShowDestModal(true);
  };

  const handleDestFormChange = (e) => {
    const { name, value, type, checked } = e.target;
    setDestForm(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleDestSubmit = async (e) => {
    e.preventDefault();
    try {
      const submitData = {
        ...destForm,
        highlights: destForm.highlights.split(',').map(h => h.trim()).filter(Boolean),
        popularity: parseInt(destForm.popularity),
        yearlyArrivals: parseInt(destForm.yearlyArrivals),
        averageStayDays: parseFloat(destForm.averageStayDays)
      };

      if (editingDest) {
        await destinationService.updateDestination(editingDest._id, submitData);
        showAlert('success', 'Destination updated successfully');
      } else {
        await destinationService.createDestination(submitData);
        showAlert('success', 'Destination created successfully');
      }
      setShowDestModal(false);
      fetchDestinations();
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Operation failed');
    }
  };

  const handleDeleteDest = async (destId) => {
    if (!confirm('Are you sure you want to delete this destination?')) return;
    try {
      await destinationService.deleteDestination(destId);
      showAlert('success', 'Destination deleted successfully');
      fetchDestinations();
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Delete failed');
    }
  };

  const handleToggleFeatured = async (destId) => {
    try {
      await destinationService.toggleFeatured(destId);
      showAlert('success', 'Featured status updated');
      fetchDestinations();
    } catch (error) {
      showAlert('error', 'Failed to update featured status');
    }
  };

  // Hotel Management Functions
  const openAddHotelModal = (dest) => {
    setCurrentDestForHotel(dest);
    setHotelForm({
      name: '',
      rating: 3,
      priceRange: 'mid-range',
      pricePerNight: 0,
      amenities: '',
      contact: ''
    });
    setShowHotelModal(true);
  };

  const handleHotelFormChange = (e) => {
    const { name, value } = e.target;
    setHotelForm(prev => ({ ...prev, [name]: value }));
  };

  const handleHotelSubmit = async (e) => {
    e.preventDefault();
    try {
      const submitData = {
        ...hotelForm,
        rating: parseInt(hotelForm.rating),
        pricePerNight: parseInt(hotelForm.pricePerNight),
        amenities: hotelForm.amenities.split(',').map(a => a.trim()).filter(Boolean)
      };
      await destinationService.addHotel(currentDestForHotel._id, submitData);
      showAlert('success', 'Hotel added successfully');
      setShowHotelModal(false);
      fetchDestinations();
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Failed to add hotel');
    }
  };

  const handleDeleteHotel = async (destId, hotelId) => {
    if (!confirm('Delete this hotel?')) return;
    try {
      await destinationService.deleteHotel(destId, hotelId);
      showAlert('success', 'Hotel deleted');
      fetchDestinations();
    } catch (error) {
      showAlert('error', 'Failed to delete hotel');
    }
  };

  // Flight Management Functions
  const openAddFlightModal = (dest) => {
    setCurrentDestForFlight(dest);
    setFlightForm({
      airline: '',
      from: 'Colombo',
      price: 0,
      duration: '',
      frequency: '',
      isEconomical: false
    });
    setShowFlightModal(true);
  };

  const handleFlightFormChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFlightForm(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleFlightSubmit = async (e) => {
    e.preventDefault();
    try {
      const submitData = {
        ...flightForm,
        price: parseInt(flightForm.price)
      };
      await destinationService.addFlight(currentDestForFlight._id, submitData);
      showAlert('success', 'Flight route added successfully');
      setShowFlightModal(false);
      fetchDestinations();
    } catch (error) {
      showAlert('error', error.response?.data?.message || 'Failed to add flight');
    }
  };

  const handleDeleteFlight = async (destId, flightId) => {
    if (!confirm('Delete this flight route?')) return;
    try {
      await destinationService.deleteFlight(destId, flightId);
      showAlert('success', 'Flight route deleted');
      fetchDestinations();
    } catch (error) {
      showAlert('error', 'Failed to delete flight');
    }
  };

  if (loading) {
    return <Loading message="Loading admin panel..." />;
  }

  return (
    <div className="admin-panel">
      {/* Header */}
      <div className="admin-header">
        <div className="admin-header-content">
          <h1 className="admin-title">Admin Panel</h1>
          <p className="admin-subtitle">Manage files and users for the Tourist Prediction System</p>
        </div>
      </div>

      {/* Alert */}
      {alert && (
        <div className={`alert ${alert.type}`}>
          {alert.type === 'success' ? '✅' : '❌'} {alert.message}
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="admin-stats-grid">
          <div className="admin-stat-card">
            <div className="admin-stat-icon">👥</div>
            <div className="admin-stat-info">
              <h3>Total Users</h3>
              <p>{stats.totalUsers}</p>
            </div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-icon">👑</div>
            <div className="admin-stat-info">
              <h3>Admins</h3>
              <p>{stats.totalAdmins}</p>
            </div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-icon">📁</div>
            <div className="admin-stat-info">
              <h3>Total Files</h3>
              <p>{stats.totalFiles}</p>
            </div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-icon">✅</div>
            <div className="admin-stat-info">
              <h3>Active Files</h3>
              <p>{stats.activeFiles}</p>
            </div>
          </div>
          <div className="admin-stat-card">
            <div className="admin-stat-icon">📥</div>
            <div className="admin-stat-info">
              <h3>Downloads</h3>
              <p>{stats.totalDownloads}</p>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="admin-tabs">
        <button
          className={`admin-tab ${activeTab === 'files' ? 'active' : ''}`}
          onClick={() => setActiveTab('files')}
        >
          📁 Files
        </button>
        <button
          className={`admin-tab ${activeTab === 'users' ? 'active' : ''}`}
          onClick={() => setActiveTab('users')}
        >
          👥 Users
        </button>
        <button
          className={`admin-tab ${activeTab === 'destinations' ? 'active' : ''}`}
          onClick={() => setActiveTab('destinations')}
        >
          🏝️ Destinations
        </button>
        <button
          className={`admin-tab ${activeTab === 'csv-predict' ? 'active' : ''}`}
          onClick={() => setActiveTab('csv-predict')}
        >
          📊 CSV Predict
        </button>
      </div>

      {/* File Management Tab */}
      {activeTab === 'files' && (
        <>
          {/* Upload Section */}
          <div className="admin-section">
            <h2>📤 Upload CSV File</h2>
            <form className="upload-form" onSubmit={handleUpload}>
              <div className={`file-input-wrapper ${selectedFile ? 'has-file' : ''}`}>
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileSelect}
                />
                <div className="file-input-label">
                  <span>{selectedFile ? '✅' : '📂'}</span>
                  <span>
                    {selectedFile 
                      ? `Selected: ${selectedFile.name} (${formatFileSize(selectedFile.size)})`
                      : 'Click or drag CSV file here to upload'
                    }
                  </span>
                </div>
              </div>
              <input
                type="text"
                className="description-input"
                placeholder="File description (optional)"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
              <button 
                type="submit" 
                className="upload-btn"
                disabled={!selectedFile || uploading}
              >
                {uploading ? '⏳ Uploading...' : '📤 Upload File'}
              </button>
            </form>
          </div>

          {/* Files List */}
          <div className="admin-section">
            <h2>📋 Uploaded Files</h2>
            {files.length === 0 ? (
              <div className="empty-state">
                <span>📭</span>
                <p>No files uploaded yet</p>
              </div>
            ) : (
              <div className="files-table-container">
                <table className="files-table">
                  <thead>
                    <tr>
                      <th>File Name</th>
                      <th>Description</th>
                      <th>Size</th>
                      <th>Downloads</th>
                      <th>Status</th>
                      <th>Uploaded</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {files.map((file) => (
                      <tr key={file._id}>
                        <td>{file.originalName}</td>
                        <td>{file.description || '-'}</td>
                        <td>{formatFileSize(file.fileSize)}</td>
                        <td>{file.downloadCount}</td>
                        <td>
                          <span className={`file-status ${file.isActive ? 'active' : 'inactive'}`}>
                            {file.isActive ? 'Active' : 'Inactive'}
                          </span>
                        </td>
                        <td>{formatDate(file.createdAt)}</td>
                        <td>
                          <div className="file-actions">
                            <button
                              className="action-btn download"
                              onClick={() => handleDownload(file)}
                              title="Download"
                            >
                              📥
                            </button>
                            <button
                              className="action-btn toggle"
                              onClick={() => handleToggleStatus(file._id)}
                              title={file.isActive ? 'Deactivate' : 'Activate'}
                            >
                              {file.isActive ? '🔒' : '🔓'}
                            </button>
                            <button
                              className="action-btn delete"
                              onClick={() => handleDeleteFile(file._id)}
                              title="Delete"
                            >
                              🗑️
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}

      {/* User Management Tab */}
      {activeTab === 'users' && (
        <div className="admin-section">
          <div className="section-header">
            <h2>👥 User Management</h2>
            <button className="create-btn" onClick={openCreateUserModal}>
              ➕ Add User
            </button>
          </div>
          {users.length === 0 ? (
            <div className="empty-state">
              <span>👤</span>
              <p>No users found</p>
            </div>
          ) : (
            <div className="files-table-container">
              <table className="users-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Email</th>
                    <th>Organization</th>
                    <th>Role</th>
                    <th>Joined</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr key={u._id}>
                      <td>{u.name}</td>
                      <td>{u.email}</td>
                      <td>{u.organization || '-'}</td>
                      <td>
                        <span className={`user-role ${u.role}`}>
                          {u.role === 'admin' ? '👑 Admin' : '👤 User'}
                        </span>
                      </td>
                      <td>{formatDate(u.createdAt)}</td>
                      <td>
                        <div className="user-actions">
                          <button
                            className="action-btn edit"
                            onClick={() => openEditUserModal(u)}
                            title="Edit"
                          >
                            ✏️
                          </button>
                          {u._id !== user?._id && (
                            <>
                              <select
                                className="role-select-mini"
                                value={u.role}
                                onChange={(e) => handleRoleChange(u._id, e.target.value)}
                              >
                                <option value="user">User</option>
                                <option value="admin">Admin</option>
                              </select>
                              <button
                                className="action-btn delete"
                                onClick={() => handleDeleteUser(u._id)}
                                title="Delete"
                              >
                                🗑️
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* CSV Predictions Tab */}
      {activeTab === 'csv-predict' && (
        <div className="admin-section">
          <h2>📊 CSV Batch Predictions</h2>
          <p className="section-description">
            Upload a CSV file with columns: year, month, dollarRate, predictionType
          </p>
          
          <form className="upload-form csv-predict-form" onSubmit={handleCsvUpload}>
            <div className={`file-input-wrapper ${csvFile ? 'has-file' : ''}`}>
              <input
                type="file"
                accept=".csv"
                onChange={handleCsvFileSelect}
              />
              <div className="file-input-label">
                <span>{csvFile ? '✅' : '📂'}</span>
                <span>
                  {csvFile 
                    ? `Selected: ${csvFile.name} (${formatFileSize(csvFile.size)})`
                    : 'Click or drag CSV file here'
                  }
                </span>
              </div>
            </div>
            <button 
              type="submit" 
              className="upload-btn"
              disabled={!csvFile || uploadingCsv}
            >
              {uploadingCsv ? '⏳ Processing...' : '🔮 Process CSV for Predictions'}
            </button>
          </form>

          {csvResult && (
            <div className="csv-result">
              <h3>📋 CSV Processing Results</h3>
              <div className="csv-stats">
                <div className="csv-stat">
                  <span className="csv-stat-label">Total Rows:</span>
                  <span className="csv-stat-value">{csvResult.totalRows}</span>
                </div>
                <div className="csv-stat">
                  <span className="csv-stat-label">Valid Rows:</span>
                  <span className="csv-stat-value success">{csvResult.validRows}</span>
                </div>
                {csvResult.errors?.length > 0 && (
                  <div className="csv-stat">
                    <span className="csv-stat-label">Errors:</span>
                    <span className="csv-stat-value error">{csvResult.errors.length}</span>
                  </div>
                )}
              </div>
              
              {csvResult.predictionData?.length > 0 && (
                <div className="csv-preview">
                  <h4>Preview Data (First 5 rows)</h4>
                  <table className="preview-table">
                    <thead>
                      <tr>
                        <th>Year</th>
                        <th>Month</th>
                        <th>Dollar Rate</th>
                        <th>Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {csvResult.predictionData.slice(0, 5).map((row, idx) => (
                        <tr key={idx}>
                          <td>{row.year}</td>
                          <td>{row.month}</td>
                          <td>{row.dollarRate}</td>
                          <td>{row.predictionType}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {csvResult.errors?.length > 0 && (
                <div className="csv-errors">
                  <h4>⚠️ Errors</h4>
                  <ul>
                    {csvResult.errors.map((err, idx) => (
                      <li key={idx}>Row {err.row}: {err.error}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Destinations Management Tab */}
      {activeTab === 'destinations' && (
        <div className="admin-section">
          <div className="section-header">
            <h2>🏝️ Destination Management</h2>
            <button className="create-btn" onClick={openCreateDestModal}>
              ➕ Add Destination
            </button>
          </div>
          
          {destinations.length === 0 ? (
            <div className="empty-state">
              <span>🏝️</span>
              <p>No destinations found. Run the destination seeder or add destinations manually.</p>
            </div>
          ) : (
            <div className="destinations-list">
              {destinations.map((dest) => (
                <div key={dest._id} className="dest-admin-card">
                  <div className="dest-admin-header">
                    <div className="dest-admin-title">
                      <h3>{dest.name}</h3>
                      <div className="dest-badges">
                        <span className={`category-badge ${dest.category}`}>{dest.category}</span>
                        <span className="region-badge">{dest.region}</span>
                        {dest.isFeatured && <span className="featured-badge">⭐ Featured</span>}
                      </div>
                    </div>
                    <div className="dest-admin-actions">
                      <button
                        className="action-btn star"
                        onClick={() => handleToggleFeatured(dest._id)}
                        title={dest.isFeatured ? 'Remove from featured' : 'Mark as featured'}
                      >
                        {dest.isFeatured ? '⭐' : '☆'}
                      </button>
                      <button
                        className="action-btn edit"
                        onClick={() => openEditDestModal(dest)}
                        title="Edit"
                      >
                        ✏️
                      </button>
                      <button
                        className="action-btn delete"
                        onClick={() => handleDeleteDest(dest._id)}
                        title="Delete"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                  
                  <p className="dest-description">{dest.description?.substring(0, 150)}...</p>
                  
                  <div className="dest-stats-row">
                    <span>👥 {dest.yearlyArrivals?.toLocaleString()} arrivals/year</span>
                    <span>📊 {dest.popularity}% popularity</span>
                    <span>📅 {dest.averageStayDays} days avg stay</span>
                  </div>

                  {/* Hotels Section */}
                  <div className="dest-sub-section">
                    <div className="sub-section-header">
                      <h4>🏨 Hotels ({dest.hotels?.length || 0})</h4>
                      <button className="add-sub-btn" onClick={() => openAddHotelModal(dest)}>
                        ➕ Add Hotel
                      </button>
                    </div>
                    {dest.hotels?.length > 0 && (
                      <div className="sub-items-grid">
                        {dest.hotels.map((hotel) => (
                          <div key={hotel._id} className="sub-item-card">
                            <div className="sub-item-header">
                              <span>{hotel.name}</span>
                              <button 
                                className="mini-delete-btn"
                                onClick={() => handleDeleteHotel(dest._id, hotel._id)}
                              >
                                ✕
                              </button>
                            </div>
                            <div className="sub-item-details">
                              <span>{'⭐'.repeat(hotel.rating || 0)}</span>
                              <span className={`price-tag ${hotel.priceRange}`}>{hotel.priceRange}</span>
                              {hotel.pricePerNight && <span>${hotel.pricePerNight}/night</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Flights Section */}
                  <div className="dest-sub-section">
                    <div className="sub-section-header">
                      <h4>✈️ Travel Routes ({dest.flights?.length || 0})</h4>
                      <button className="add-sub-btn" onClick={() => openAddFlightModal(dest)}>
                        ➕ Add Route
                      </button>
                    </div>
                    {dest.flights?.length > 0 && (
                      <div className="sub-items-grid">
                        {dest.flights.map((flight) => (
                          <div key={flight._id} className="sub-item-card">
                            <div className="sub-item-header">
                              <span>{flight.airline}</span>
                              <button 
                                className="mini-delete-btn"
                                onClick={() => handleDeleteFlight(dest._id, flight._id)}
                              >
                                ✕
                              </button>
                            </div>
                            <div className="sub-item-details">
                              <span>From: {flight.from}</span>
                              <span>{flight.duration}</span>
                              <span className="price-tag">${flight.price}</span>
                              {flight.isEconomical && <span className="eco-badge">💰 Budget</span>}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* User Modal */}
      <Modal isOpen={showUserModal} onClose={() => setShowUserModal(false)} title={editingUser ? '✏️ Edit User' : '➕ Create User'}>
        <div className="user-modal">
          <form onSubmit={handleUserSubmit}>
            <div className="form-group">
              <label>Name</label>
              <input
                type="text"
                name="name"
                value={userForm.name}
                onChange={handleUserFormChange}
                required
              />
            </div>
            <div className="form-group">
              <label>Email</label>
              <input
                type="email"
                name="email"
                value={userForm.email}
                onChange={handleUserFormChange}
                required
              />
            </div>
            <div className="form-group">
              <label>{editingUser ? 'New Password (leave blank to keep current)' : 'Password'}</label>
              <input
                type="password"
                name="password"
                value={userForm.password}
                onChange={handleUserFormChange}
                required={!editingUser}
                minLength={6}
              />
            </div>
            <div className="form-group">
              <label>Role</label>
              <select
                name="role"
                value={userForm.role}
                onChange={handleUserFormChange}
              >
                <option value="user">User</option>
                <option value="admin">Admin</option>
              </select>
            </div>
            <div className="form-group">
              <label>Organization</label>
              <input
                type="text"
                name="organization"
                value={userForm.organization}
                onChange={handleUserFormChange}
              />
            </div>
            <div className="modal-actions">
              <button type="button" className="cancel-btn" onClick={() => setShowUserModal(false)}>
                Cancel
              </button>
              <button type="submit" className="submit-btn">
                {editingUser ? 'Update' : 'Create'}
              </button>
            </div>
          </form>
        </div>
      </Modal>

      {/* Destination Modal */}
      <Modal isOpen={showDestModal} onClose={() => setShowDestModal(false)} title={editingDest ? '✏️ Edit Destination' : '➕ Create Destination'}>
        <div className="dest-modal">
          <form onSubmit={handleDestSubmit}>
            <div className="form-row">
              <div className="form-group">
                <label>Name *</label>
                <input
                  type="text"
                  name="name"
                  value={destForm.name}
                  onChange={handleDestFormChange}
                  required
                  placeholder="e.g., Sigiriya"
                />
              </div>
              <div className="form-group">
                <label>Region *</label>
                <select name="region" value={destForm.region} onChange={handleDestFormChange} required>
                  {REGIONS.map(r => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
            </div>
            
            <div className="form-group">
              <label>Description *</label>
              <textarea
                name="description"
                value={destForm.description}
                onChange={handleDestFormChange}
                required
                rows={3}
                placeholder="Describe the destination..."
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Category *</label>
                <select name="category" value={destForm.category} onChange={handleDestFormChange} required>
                  {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Best Time to Visit</label>
                <input
                  type="text"
                  name="bestTimeToVisit"
                  value={destForm.bestTimeToVisit}
                  onChange={handleDestFormChange}
                  placeholder="e.g., January - April"
                />
              </div>
            </div>

            <div className="form-group">
              <label>Highlights (comma-separated)</label>
              <input
                type="text"
                name="highlights"
                value={destForm.highlights}
                onChange={handleDestFormChange}
                placeholder="e.g., Lion Rock, Frescoes, Water Gardens"
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Popularity (0-100)</label>
                <input
                  type="number"
                  name="popularity"
                  value={destForm.popularity}
                  onChange={handleDestFormChange}
                  min="0"
                  max="100"
                />
              </div>
              <div className="form-group">
                <label>Yearly Arrivals</label>
                <input
                  type="number"
                  name="yearlyArrivals"
                  value={destForm.yearlyArrivals}
                  onChange={handleDestFormChange}
                  min="0"
                />
              </div>
              <div className="form-group">
                <label>Avg Stay (days)</label>
                <input
                  type="number"
                  name="averageStayDays"
                  value={destForm.averageStayDays}
                  onChange={handleDestFormChange}
                  min="1"
                  step="0.5"
                />
              </div>
            </div>

            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  name="isFeatured"
                  checked={destForm.isFeatured}
                  onChange={handleDestFormChange}
                />
                Mark as Featured
              </label>
            </div>

            <div className="modal-actions">
              <button type="button" className="cancel-btn" onClick={() => setShowDestModal(false)}>
                Cancel
              </button>
              <button type="submit" className="submit-btn">
                {editingDest ? 'Update' : 'Create'}
              </button>
            </div>
          </form>
        </div>
      </Modal>

      {/* Hotel Modal */}
      <Modal isOpen={showHotelModal} onClose={() => setShowHotelModal(false)} title={`🏨 Add Hotel to ${currentDestForHotel?.name}`}>
        <div className="hotel-modal">
          <form onSubmit={handleHotelSubmit}>
            <div className="form-group">
              <label>Hotel Name *</label>
              <input
                type="text"
                name="name"
                value={hotelForm.name}
                onChange={handleHotelFormChange}
                required
                placeholder="e.g., Grand Hotel"
              />
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label>Rating (1-5)</label>
                <select name="rating" value={hotelForm.rating} onChange={handleHotelFormChange}>
                  {[1,2,3,4,5].map(r => <option key={r} value={r}>{r} Star</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Price Range</label>
                <select name="priceRange" value={hotelForm.priceRange} onChange={handleHotelFormChange}>
                  <option value="budget">Budget</option>
                  <option value="mid-range">Mid-Range</option>
                  <option value="luxury">Luxury</option>
                </select>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Price per Night (LKR)</label>
                <input
                  type="number"
                  name="pricePerNight"
                  value={hotelForm.pricePerNight}
                  onChange={handleHotelFormChange}
                  min="0"
                />
              </div>
              <div className="form-group">
                <label>Contact</label>
                <input
                  type="text"
                  name="contact"
                  value={hotelForm.contact}
                  onChange={handleHotelFormChange}
                  placeholder="Phone number"
                />
              </div>
            </div>

            <div className="form-group">
              <label>Amenities (comma-separated)</label>
              <input
                type="text"
                name="amenities"
                value={hotelForm.amenities}
                onChange={handleHotelFormChange}
                placeholder="e.g., Pool, Spa, WiFi, Restaurant"
              />
            </div>

            <div className="modal-actions">
              <button type="button" className="cancel-btn" onClick={() => setShowHotelModal(false)}>
                Cancel
              </button>
              <button type="submit" className="submit-btn">Add Hotel</button>
            </div>
          </form>
        </div>
      </Modal>

      {/* Flight Modal */}
      <Modal isOpen={showFlightModal} onClose={() => setShowFlightModal(false)} title={`✈️ Add Route to ${currentDestForFlight?.name}`}>
        <div className="flight-modal">
          <form onSubmit={handleFlightSubmit}>
            <div className="form-row">
              <div className="form-group">
                <label>Transport/Airline *</label>
                <input
                  type="text"
                  name="airline"
                  value={flightForm.airline}
                  onChange={handleFlightFormChange}
                  required
                  placeholder="e.g., Sri Lankan Airlines, Bus, Train"
                />
              </div>
              <div className="form-group">
                <label>From *</label>
                <input
                  type="text"
                  name="from"
                  value={flightForm.from}
                  onChange={handleFlightFormChange}
                  required
                  placeholder="e.g., Colombo"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Duration</label>
                <input
                  type="text"
                  name="duration"
                  value={flightForm.duration}
                  onChange={handleFlightFormChange}
                  placeholder="e.g., 3.5 hours"
                />
              </div>
              <div className="form-group">
                <label>Frequency</label>
                <input
                  type="text"
                  name="frequency"
                  value={flightForm.frequency}
                  onChange={handleFlightFormChange}
                  placeholder="e.g., Daily, Hourly"
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Price (LKR)</label>
                <input
                  type="number"
                  name="price"
                  value={flightForm.price}
                  onChange={handleFlightFormChange}
                  min="0"
                />
              </div>
              <div className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    name="isEconomical"
                    checked={flightForm.isEconomical}
                    onChange={handleFlightFormChange}
                  />
                  Budget-Friendly Option
                </label>
              </div>
            </div>

            <div className="modal-actions">
              <button type="button" className="cancel-btn" onClick={() => setShowFlightModal(false)}>
                Cancel
              </button>
              <button type="submit" className="submit-btn">Add Route</button>
            </div>
          </form>
        </div>
      </Modal>
    </div>
  );
}

export default AdminPage;
