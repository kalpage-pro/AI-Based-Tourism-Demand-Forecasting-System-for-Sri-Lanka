import { useState, useEffect } from 'react';
import { 
  exportPredictionsCSV, 
  exportPredictionsPDF, 
  exportHistoricalData,
  getExportHistory 
} from '../services/export.service';
import Loading from '../components/Common/Loading';
import Card from '../components/Common/Card';
import Button from '../components/Common/Button';
import './ReportsPage.css';

function ReportsPage() {
  const [loading, setLoading] = useState(false);
  const [exportHistory, setExportHistory] = useState([]);
  const [exportType, setExportType] = useState('predictions');
  const [format, setFormat] = useState('csv');
  const [dateRange, setDateRange] = useState({
    startDate: '',
    endDate: ''
  });
  const [predictionType, setPredictionType] = useState('all');
  const [message, setMessage] = useState(null);

  useEffect(() => {
    loadExportHistory();
  }, []);

  const loadExportHistory = async () => {
    try {
      const response = await getExportHistory();
      setExportHistory(response.data || []);
    } catch (err) {
      console.error('Failed to load export history:', err);
    }
  };

  const handleExport = async () => {
    try {
      setLoading(true);
      setMessage(null);

      const filters = {
        ...dateRange,
        predictionType
      };

      if (exportType === 'predictions') {
        if (format === 'csv') {
          await exportPredictionsCSV(filters);
        } else {
          await exportPredictionsPDF(filters);
        }
      } else {
        await exportHistoricalData(format, filters);
      }

      setMessage({ type: 'success', text: 'Export completed successfully!' });
      loadExportHistory();
    } catch (err) {
      setMessage({ type: 'error', text: 'Export failed. Please try again.' });
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="reports-page">
      <div className="reports-header">
        <h1>📑 Reports & Exports</h1>
        <p>Generate and download reports from your tourism data</p>
      </div>

      <div className="reports-content">
        <Card className="export-card">
          <h3>📊 Export Data</h3>
          
          <div className="export-options">
            <div className="option-group">
              <label>Export Type</label>
              <div className="radio-buttons">
                <label className={`radio-btn ${exportType === 'predictions' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="exportType"
                    value="predictions"
                    checked={exportType === 'predictions'}
                    onChange={(e) => setExportType(e.target.value)}
                  />
                  <span>📈 My Predictions</span>
                </label>
                <label className={`radio-btn ${exportType === 'historical' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="exportType"
                    value="historical"
                    checked={exportType === 'historical'}
                    onChange={(e) => setExportType(e.target.value)}
                  />
                  <span>📚 Historical Data</span>
                </label>
              </div>
            </div>

            <div className="option-group">
              <label>Export Format</label>
              <div className="radio-buttons">
                <label className={`radio-btn ${format === 'csv' ? 'active' : ''}`}>
                  <input
                    type="radio"
                    name="format"
                    value="csv"
                    checked={format === 'csv'}
                    onChange={(e) => setFormat(e.target.value)}
                  />
                  <span>📄 CSV</span>
                </label>
                {exportType === 'predictions' && (
                  <label className={`radio-btn ${format === 'pdf' ? 'active' : ''}`}>
                    <input
                      type="radio"
                      name="format"
                      value="pdf"
                      checked={format === 'pdf'}
                      onChange={(e) => setFormat(e.target.value)}
                    />
                    <span>📕 PDF Report</span>
                  </label>
                )}
              </div>
            </div>

            {exportType === 'predictions' && (
              <>
                <div className="option-group">
                  <label>Date Range (Optional)</label>
                  <div className="date-inputs">
                    <input
                      type="date"
                      value={dateRange.startDate}
                      onChange={(e) => setDateRange({...dateRange, startDate: e.target.value})}
                      placeholder="Start Date"
                    />
                    <span>to</span>
                    <input
                      type="date"
                      value={dateRange.endDate}
                      onChange={(e) => setDateRange({...dateRange, endDate: e.target.value})}
                      placeholder="End Date"
                    />
                  </div>
                </div>

                <div className="option-group">
                  <label>Prediction Type</label>
                  <select 
                    value={predictionType} 
                    onChange={(e) => setPredictionType(e.target.value)}
                  >
                    <option value="all">All Types</option>
                    <option value="tourist_arrivals">Tourist Arrivals Only</option>
                    <option value="revenue">Revenue Only</option>
                    <option value="rooms">Occupancy Only</option>
                  </select>
                </div>
              </>
            )}
          </div>

          {message && (
            <div className={`message ${message.type}`}>
              {message.text}
            </div>
          )}

          <Button 
            onClick={handleExport} 
            disabled={loading}
            className="export-btn"
          >
            {loading ? (
              '⏳ Exporting...'
            ) : (
              <>
                📥 Export {format.toUpperCase()}
              </>
            )}
          </Button>
        </Card>

        <Card className="templates-card">
          <h3>📋 Quick Export Templates</h3>
          
          <div className="templates-list">
            <div 
              className="template-item"
              onClick={() => {
                setExportType('predictions');
                setFormat('pdf');
                setTimeout(handleExport, 100);
              }}
            >
              <div className="template-icon">📊</div>
              <div className="template-info">
                <h4>Full Prediction Report</h4>
                <p>PDF report with all your predictions and statistics</p>
              </div>
            </div>

            <div 
              className="template-item"
              onClick={() => {
                setExportType('predictions');
                setFormat('csv');
                setTimeout(handleExport, 100);
              }}
            >
              <div className="template-icon">📈</div>
              <div className="template-info">
                <h4>Prediction Data Export</h4>
                <p>CSV file with all prediction data for analysis</p>
              </div>
            </div>

            <div 
              className="template-item"
              onClick={() => {
                setExportType('historical');
                setFormat('csv');
                setTimeout(handleExport, 100);
              }}
            >
              <div className="template-icon">📚</div>
              <div className="template-info">
                <h4>Historical Tourism Data</h4>
                <p>CSV export of Sri Lanka tourism historical records</p>
              </div>
            </div>
          </div>
        </Card>

        <Card className="history-card">
          <h3>📜 Recent Exports</h3>
          
          {exportHistory.length > 0 ? (
            <div className="history-list">
              {exportHistory.map((item, idx) => (
                <div key={idx} className="history-item">
                  <div className="history-icon">
                    {item.details?.format === 'pdf' ? '📕' : '📄'}
                  </div>
                  <div className="history-info">
                    <span className="history-format">
                      {item.details?.format?.toUpperCase() || 'Export'}
                    </span>
                    <span className="history-count">
                      {item.details?.count} records
                    </span>
                  </div>
                  <div className="history-date">
                    {new Date(item.createdAt).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="no-history">No export history yet. Start by exporting some data!</p>
          )}
        </Card>
      </div>
    </div>
  );
}

export default ReportsPage;
