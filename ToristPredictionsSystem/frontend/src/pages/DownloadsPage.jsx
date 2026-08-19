import { useState, useEffect } from 'react';
import adminService from '../services/admin.service';
import Loading from '../components/Common/Loading';
import './DownloadsPage.css';

function DownloadsPage() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(null);

  useEffect(() => {
    fetchFiles();
  }, []);

  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await adminService.getAvailableFiles();
      setFiles(response.data);
    } catch (error) {
      console.error('Failed to fetch files:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (file) => {
    try {
      setDownloading(file._id);
      await adminService.downloadFile(file._id, file.originalName);
    } catch (error) {
      console.error('Download failed:', error);
      alert('Failed to download file. Please try again.');
    } finally {
      setDownloading(null);
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
      day: 'numeric'
    });
  };

  if (loading) {
    return <Loading message="Loading available files..." />;
  }

  return (
    <div className="downloads-page">
      {/* Header */}
      <div className="downloads-header">
        <div className="downloads-header-content">
          <h1 className="downloads-title">📥 Downloads</h1>
          <p className="downloads-subtitle">Download CSV data files shared by administrators</p>
        </div>
      </div>

      {/* Files */}
      {files.length === 0 ? (
        <div className="downloads-empty">
          <span>📭</span>
          <h3>No Files Available</h3>
          <p>There are no files available for download at the moment. Check back later!</p>
        </div>
      ) : (
        <div className="files-grid">
          {files.map((file) => (
            <div key={file._id} className="file-card">
              <div className="file-card-header">
                <div className="file-icon">📄</div>
                <div className="file-info">
                  <h3>{file.originalName}</h3>
                  <p>{formatFileSize(file.fileSize)}</p>
                </div>
              </div>
              
              <div className="file-description">
                {file.description || 'No description provided.'}
              </div>
              
              <div className="file-meta">
                <span>📅 {formatDate(file.createdAt)}</span>
                <span>📥 {file.downloadCount} downloads</span>
              </div>
              
              <button 
                className="download-btn"
                onClick={() => handleDownload(file)}
                disabled={downloading === file._id}
              >
                {downloading === file._id ? (
                  <>⏳ Downloading...</>
                ) : (
                  <>📥 Download File</>
                )}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default DownloadsPage;
