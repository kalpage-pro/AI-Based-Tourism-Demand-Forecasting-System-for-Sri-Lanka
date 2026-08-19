import { useState, useEffect } from 'react';
import {
  getFeatureImportance,
  compareModels,
  getSeasonalPatterns,
  getYearlyTrends,
  getForecast,
  getTourismDashboard
} from '../services/analytics.service';
import {
  getFeaturedDestinations,
  getEconomicalFlights,
  getBestHotels
} from '../services/destination.service';
import Loading from '../components/Common/Loading';
import Card from '../components/Common/Card';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, RadarChart, Radar, 
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, AreaChart, Area,
  PieChart, Pie, Cell
} from 'recharts';
import './AnalyticsPage.css';

// Color palette for charts
const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
const RADIAN = Math.PI / 180;

// Custom label for pie chart
const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x = cx + radius * Math.cos(-midAngle * RADIAN);
  const y = cy + radius * Math.sin(-midAngle * RADIAN);

  return percent > 0.05 ? (
    <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central" fontSize={12}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  ) : null;
};

function AnalyticsPage() {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('destinations');
  const [featureImportance, setFeatureImportance] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [seasonalPatterns, setSeasonalPatterns] = useState(null);
  const [yearlyTrends, setYearlyTrends] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [selectedModel, setSelectedModel] = useState('rf');
  const [error, setError] = useState(null);
  
  // New state for destinations
  const [tourismDashboard, setTourismDashboard] = useState(null);
  const [featuredDestinations, setFeaturedDestinations] = useState([]);
  const [economicalFlights, setEconomicalFlights] = useState([]);
  const [bestHotels, setBestHotels] = useState([]);
  const [hotelFilter, setHotelFilter] = useState('all');

  useEffect(() => {
    loadAnalyticsData();
  }, []);

  const loadAnalyticsData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [fiRes, mcRes, spRes, ytRes, tdRes, fdRes, efRes, bhRes] = await Promise.all([
        getFeatureImportance('all').catch(() => ({ data: null })),
        compareModels().catch(() => ({ data: null })),
        getSeasonalPatterns().catch(() => ({ data: null })),
        getYearlyTrends().catch(() => ({ data: null })),
        getTourismDashboard().catch(() => ({ data: null })),
        getFeaturedDestinations().catch(() => ({ data: [] })),
        getEconomicalFlights().catch(() => ({ data: [] })),
        getBestHotels().catch(() => ({ data: [] }))
      ]);

      setFeatureImportance(fiRes.data);
      setModelComparison(mcRes.data);
      setSeasonalPatterns(spRes.data);
      setYearlyTrends(ytRes.data);
      setTourismDashboard(tdRes.data);
      setFeaturedDestinations(fdRes.data || []);
      setEconomicalFlights(efRes.data || []);
      setBestHotels(bhRes.data || []);

      // Load forecast
      const currentYear = new Date().getFullYear();
      const currentMonth = new Date().getMonth() + 1;
      const fcRes = await getForecast(currentYear, currentMonth, selectedModel).catch(() => ({ data: null }));
      setForecast(fcRes.data);

    } catch (err) {
      setError('Failed to load analytics data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadForecast = async (model) => {
    setSelectedModel(model);
    const currentYear = new Date().getFullYear();
    const currentMonth = new Date().getMonth() + 1;
    try {
      const fcRes = await getForecast(currentYear, currentMonth, model);
      setForecast(fcRes.data);
    } catch (err) {
      console.error('Failed to load forecast:', err);
    }
  };

  if (loading) {
    return <Loading message="Loading analytics data..." />;
  }

  return (
    <div className="analytics-page">
      <div className="analytics-header">
        <h1>📊 Tourism Analytics</h1>
        <p>Advanced insights, destinations, and model performance analysis</p>
      </div>

      <div className="analytics-tabs">
        <button 
          className={`tab-btn ${activeTab === 'destinations' ? 'active' : ''}`}
          onClick={() => setActiveTab('destinations')}
        >
          🏝️ Destinations
        </button>
        <button 
          className={`tab-btn ${activeTab === 'arrivals' ? 'active' : ''}`}
          onClick={() => setActiveTab('arrivals')}
        >
          📈 Arrivals
        </button>
        <button 
          className={`tab-btn ${activeTab === 'hotels' ? 'active' : ''}`}
          onClick={() => setActiveTab('hotels')}
        >
          🏨 Hotels
        </button>
        <button 
          className={`tab-btn ${activeTab === 'flights' ? 'active' : ''}`}
          onClick={() => setActiveTab('flights')}
        >
          ✈️ Flights
        </button>
        <button 
          className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        <button 
          className={`tab-btn ${activeTab === 'features' ? 'active' : ''}`}
          onClick={() => setActiveTab('features')}
        >
          Features
        </button>
        <button 
          className={`tab-btn ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => setActiveTab('models')}
        >
          Model Comparison
        </button>
        <button 
          className={`tab-btn ${activeTab === 'seasonal' ? 'active' : ''}`}
          onClick={() => setActiveTab('seasonal')}
        >
          Seasonal Patterns
        </button>
        <button 
          className={`tab-btn ${activeTab === 'forecast' ? 'active' : ''}`}
          onClick={() => setActiveTab('forecast')}
        >
          12-Month Forecast
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="analytics-content">
        {/* Destinations Tab */}
        {activeTab === 'destinations' && (
          <div className="destinations-section">
            {/* Overview Stats */}
            {tourismDashboard?.overview && (
              <div className="destination-stats-grid">
                <Card className="stat-card highlight">
                  <div className="stat-icon">🏝️</div>
                  <div className="stat-info">
                    <h4>Total Destinations</h4>
                    <div className="stat-value">{tourismDashboard.overview.totalDestinations}</div>
                  </div>
                </Card>
                <Card className="stat-card highlight">
                  <div className="stat-icon">👥</div>
                  <div className="stat-info">
                    <h4>Yearly Arrivals</h4>
                    <div className="stat-value">{tourismDashboard.overview.totalYearlyArrivals?.toLocaleString()}</div>
                  </div>
                </Card>
                <Card className="stat-card highlight">
                  <div className="stat-icon">⭐</div>
                  <div className="stat-info">
                    <h4>Avg Rating</h4>
                    <div className="stat-value">{tourismDashboard.overview.avgRating?.toFixed(1) || 'N/A'}</div>
                  </div>
                </Card>
                <Card className="stat-card highlight">
                  <div className="stat-icon">📅</div>
                  <div className="stat-info">
                    <h4>Avg Stay</h4>
                    <div className="stat-value">{tourismDashboard.overview.avgStayDays?.toFixed(1)} days</div>
                  </div>
                </Card>
              </div>
            )}

            {/* Pie Charts Row */}
            <div className="pie-charts-row">
              {/* Category Pie Chart */}
              <Card className="chart-card">
                <h3>🎯 Arrivals by Category</h3>
                {tourismDashboard?.charts?.categoryPieData?.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={tourismDashboard.charts.categoryPieData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={renderCustomizedLabel}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {tourismDashboard.charts.categoryPieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => value?.toLocaleString()} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="no-data">No category data available</p>
                )}
              </Card>

              {/* Region Pie Chart */}
              <Card className="chart-card">
                <h3>🗺️ Arrivals by Region</h3>
                {tourismDashboard?.charts?.regionPieData?.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={tourismDashboard.charts.regionPieData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={renderCustomizedLabel}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {tourismDashboard.charts.regionPieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => value?.toLocaleString()} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="no-data">No region data available</p>
                )}
              </Card>
            </div>

            {/* Featured Destinations */}
            <Card className="chart-card full-width">
              <h3>⭐ Featured Destinations</h3>
              {featuredDestinations.length > 0 ? (
                <div className="featured-destinations-grid">
                  {featuredDestinations.map((dest, idx) => (
                    <div key={idx} className="featured-destination-card">
                      <div className="dest-header">
                        <h4>{dest.name}</h4>
                        <span className={`category-badge ${dest.category}`}>{dest.category}</span>
                      </div>
                      <p className="dest-region">📍 {dest.region}</p>
                      <p className="dest-description">{dest.description?.substring(0, 100)}...</p>
                      <div className="dest-stats">
                        <span>👥 {dest.yearlyArrivals?.toLocaleString()} arrivals/year</span>
                        <span>⭐ {dest.ratings?.overall?.toFixed(1) || 'N/A'}</span>
                      </div>
                      {dest.highlights && dest.highlights.length > 0 && (
                        <div className="dest-highlights">
                          {dest.highlights.slice(0, 3).map((h, i) => (
                            <span key={i} className="highlight-tag">{h}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-data">No featured destinations. Run the destination seeder first.</p>
              )}
            </Card>

            {/* Top Destinations Bar Chart */}
            <Card className="chart-card full-width">
              <h3>🏆 Top 10 Destinations by Arrivals</h3>
              {tourismDashboard?.topDestinations?.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={tourismDashboard.topDestinations} layout="vertical" margin={{ left: 100 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis type="category" dataKey="name" width={90} />
                    <Tooltip formatter={(value) => value?.toLocaleString()} />
                    <Bar dataKey="yearlyArrivals" name="Yearly Arrivals" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="no-data">No destination data available</p>
              )}
            </Card>
          </div>
        )}

        {/* Arrivals Tab */}
        {activeTab === 'arrivals' && (
          <div className="arrivals-section">
            <div className="pie-charts-row">
              {/* Category Distribution */}
              <Card className="chart-card">
                <h3>📊 Category Distribution</h3>
                {tourismDashboard?.charts?.categoryPieData?.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={tourismDashboard.charts.categoryPieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {tourismDashboard.charts.categoryPieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => value?.toLocaleString()} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="category-legend">
                      {tourismDashboard.charts.categoryPieData.map((cat, idx) => (
                        <div key={idx} className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></span>
                          <span className="legend-name">{cat.name}</span>
                          <span className="legend-value">{cat.value?.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="no-data">No category data available</p>
                )}
              </Card>

              {/* Region Distribution */}
              <Card className="chart-card">
                <h3>🌍 Regional Distribution</h3>
                {tourismDashboard?.charts?.regionPieData?.length > 0 ? (
                  <>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={tourismDashboard.charts.regionPieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {tourismDashboard.charts.regionPieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => value?.toLocaleString()} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="category-legend">
                      {tourismDashboard.charts.regionPieData.map((reg, idx) => (
                        <div key={idx} className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: COLORS[idx % COLORS.length] }}></span>
                          <span className="legend-name">{reg.name}</span>
                          <span className="legend-value">{reg.value?.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="no-data">No region data available</p>
                )}
              </Card>
            </div>

            {/* Historical Comparison */}
            <Card className="chart-card full-width">
              <h3>📅 Historical Yearly Arrivals</h3>
              {tourismDashboard?.charts?.yearlyComparison?.length > 0 ? (
                <ResponsiveContainer width="100%" height={350}>
                  <AreaChart data={tourismDashboard.charts.yearlyComparison}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="_id" />
                    <YAxis />
                    <Tooltip formatter={(value) => value?.toLocaleString()} />
                    <Legend />
                    <Area type="monotone" dataKey="totalArrivals" name="Total Arrivals" stroke="#3b82f6" fill="#93c5fd" fillOpacity={0.6} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <p className="no-data">No historical data available</p>
              )}
            </Card>
          </div>
        )}

        {/* Hotels Tab */}
        {activeTab === 'hotels' && (
          <div className="hotels-section">
            <Card className="chart-card full-width">
              <div className="hotels-header">
                <h3>🏨 Best Hotels in Sri Lanka</h3>
                <div className="filter-buttons">
                  <button 
                    className={`filter-btn ${hotelFilter === 'all' ? 'active' : ''}`}
                    onClick={() => setHotelFilter('all')}
                  >
                    All
                  </button>
                  <button 
                    className={`filter-btn ${hotelFilter === 'luxury' ? 'active' : ''}`}
                    onClick={() => setHotelFilter('luxury')}
                  >
                    Luxury
                  </button>
                  <button 
                    className={`filter-btn ${hotelFilter === 'mid-range' ? 'active' : ''}`}
                    onClick={() => setHotelFilter('mid-range')}
                  >
                    Mid-Range
                  </button>
                  <button 
                    className={`filter-btn ${hotelFilter === 'budget' ? 'active' : ''}`}
                    onClick={() => setHotelFilter('budget')}
                  >
                    Budget
                  </button>
                </div>
              </div>
              
              {bestHotels.length > 0 ? (
                <div className="hotels-grid">
                  {bestHotels
                    .filter(h => hotelFilter === 'all' || h.priceRange === hotelFilter)
                    .map((hotel, idx) => (
                    <div key={idx} className={`hotel-card ${hotel.priceRange}`}>
                      <div className="hotel-header">
                        <h4>{hotel.name}</h4>
                        <div className="hotel-rating">
                          {'⭐'.repeat(hotel.rating || 0)}
                        </div>
                      </div>
                      <p className="hotel-location">📍 {hotel.destination}, {hotel.region}</p>
                      <div className="hotel-details">
                        <span className={`price-badge ${hotel.priceRange}`}>
                          {hotel.priceRange?.replace('-', ' ')}
                        </span>
                        {hotel.pricePerNight && (
                          <span className="hotel-price">${hotel.pricePerNight}/night</span>
                        )}
                      </div>
                      {hotel.amenities && hotel.amenities.length > 0 && (
                        <div className="hotel-amenities">
                          {hotel.amenities.slice(0, 4).map((a, i) => (
                            <span key={i} className="amenity-tag">{a}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="no-data">No hotel data available. Run the destination seeder first.</p>
              )}
            </Card>
          </div>
        )}

        {/* Flights Tab */}
        {activeTab === 'flights' && (
          <div className="flights-section">
            <Card className="chart-card full-width">
              <h3>✈️ Economical Travel Options</h3>
              {economicalFlights.length > 0 ? (
                <div className="flights-table-container">
                  <table className="flights-table">
                    <thead>
                      <tr>
                        <th>Transport</th>
                        <th>From</th>
                        <th>To</th>
                        <th>Duration</th>
                        <th>Frequency</th>
                        <th>Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {economicalFlights.map((flight, idx) => (
                        <tr key={idx}>
                          <td>
                            <span className="transport-icon">
                              {flight.airline?.toLowerCase().includes('train') ? '🚂' : 
                               flight.airline?.toLowerCase().includes('bus') ? '🚌' : '✈️'}
                            </span>
                            {flight.airline}
                          </td>
                          <td>{flight.from}</td>
                          <td>{flight.destination}</td>
                          <td>{flight.duration}</td>
                          <td>{flight.frequency}</td>
                          <td className="price-cell">${flight.price}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="no-data">No economical flight data available. Run the destination seeder first.</p>
              )}
            </Card>

            <div className="travel-tips">
              <Card className="tip-card">
                <h4>💡 Budget Travel Tips</h4>
                <ul>
                  <li>Take the scenic train from Kandy to Ella - one of the world's most beautiful train journeys!</li>
                  <li>Use intercity buses for affordable travel between major cities</li>
                  <li>Book trains in advance during peak season (Dec-Mar)</li>
                  <li>Consider shared tuk-tuk tours for short distances</li>
                </ul>
              </Card>
              <Card className="tip-card">
                <h4>🌟 Best Value Routes</h4>
                <ul>
                  <li><strong>Colombo → Galle:</strong> Coastal express bus (LKR 1,900, 2.5hrs)</li>
                  <li><strong>Kandy → Ella:</strong> Scenic train (LKR 3,200, 6hrs)</li>
                  <li><strong>Colombo → Kandy:</strong> Train (LKR 4,800, 3hrs)</li>
                  <li><strong>Colombo → Anuradhapura:</strong> Bus (LKR 2,200, 4hrs)</li>
                </ul>
              </Card>
            </div>
          </div>
        )}

        {activeTab === 'overview' && (
          <div className="overview-section">
            <div className="charts-grid">
              {/* Yearly Trends Chart */}
              <Card className="chart-card full-width">
                <h3>📈 Yearly Tourism Trends</h3>
                {yearlyTrends && yearlyTrends.length > 0 ? (
                  <ResponsiveContainer width="100%" height={350}>
                    <AreaChart data={yearlyTrends}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="year" />
                      <YAxis yAxisId="left" />
                      <YAxis yAxisId="right" orientation="right" />
                      <Tooltip formatter={(value) => value?.toLocaleString()} />
                      <Legend />
                      <Area 
                        yAxisId="left"
                        type="monotone" 
                        dataKey="totalArrivals" 
                        name="Total Arrivals"
                        stroke="#3b82f6" 
                        fill="#93c5fd"
                        fillOpacity={0.6}
                      />
                      <Line 
                        yAxisId="right"
                        type="monotone" 
                        dataKey="arrivalsGrowth" 
                        name="Growth %"
                        stroke="#10b981" 
                        strokeWidth={2}
                        dot={{ fill: '#10b981' }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="no-data">No yearly data available. Run the historical data seeder first.</p>
                )}
              </Card>

              {/* Quick Stats */}
              {yearlyTrends && yearlyTrends.length > 0 && (
                <div className="quick-stats">
                  <Card className="stat-card">
                    <h4>Latest Year</h4>
                    <div className="stat-value">{yearlyTrends[yearlyTrends.length - 1]?.year}</div>
                    <div className="stat-label">
                      {yearlyTrends[yearlyTrends.length - 1]?.totalArrivals?.toLocaleString()} arrivals
                    </div>
                  </Card>
                  <Card className="stat-card">
                    <h4>Peak Year</h4>
                    <div className="stat-value">
                      {yearlyTrends.reduce((max, y) => y.totalArrivals > max.totalArrivals ? y : max).year}
                    </div>
                    <div className="stat-label">
                      {yearlyTrends.reduce((max, y) => y.totalArrivals > max.totalArrivals ? y : max).totalArrivals?.toLocaleString()} arrivals
                    </div>
                  </Card>
                  <Card className="stat-card">
                    <h4>Average Growth</h4>
                    <div className="stat-value">
                      {(yearlyTrends.filter(y => y.arrivalsGrowth !== null)
                        .reduce((sum, y) => sum + y.arrivalsGrowth, 0) / 
                        yearlyTrends.filter(y => y.arrivalsGrowth !== null).length || 0).toFixed(1)}%
                    </div>
                    <div className="stat-label">Year-over-year</div>
                  </Card>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'features' && (
          <div className="features-section">
            <Card className="chart-card">
              <h3>🎯 Top 10 Feature Importance (Average Across Models)</h3>
              {featureImportance?.average?.features ? (
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart 
                    data={featureImportance.average.features} 
                    layout="vertical"
                    margin={{ left: 150 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" unit="%" />
                    <YAxis type="category" dataKey="description" width={140} />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="no-data">Feature importance data not available</p>
              )}
            </Card>

            {featureImportance && Object.keys(featureImportance).filter(k => k !== 'average').length > 0 && (
              <div className="model-features-grid">
                {['rf_arrivals', 'xgb_arrivals'].map(modelKey => (
                  featureImportance[modelKey]?.features && (
                    <Card key={modelKey} className="chart-card">
                      <h4>{modelKey.includes('rf') ? '🌲 Random Forest' : '⚡ XGBoost'} - Arrivals</h4>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={featureImportance[modelKey].features.slice(0, 8)} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis type="number" unit="%" />
                          <YAxis type="category" dataKey="feature" width={100} fontSize={11} />
                          <Tooltip />
                          <Bar dataKey="importance" fill={modelKey.includes('rf') ? '#10b981' : '#f59e0b'} />
                        </BarChart>
                      </ResponsiveContainer>
                    </Card>
                  )
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'models' && (
          <div className="models-section">
            <Card className="chart-card">
              <h3>🤖 Model Comparison</h3>
              {modelComparison ? (
                <div className="model-comparison-content">
                  <div className="model-cards-grid">
                    {['arrivals', 'revenue', 'occupancy'].map(target => (
                      <div key={target} className="target-section">
                        <h4 className="target-title">{target.charAt(0).toUpperCase() + target.slice(1)} Prediction</h4>
                        <div className="models-row">
                          {modelComparison[target]?.random_forest && (
                            <div className="model-info-card rf">
                              <div className="model-badge">🌲 Random Forest</div>
                              <div className="model-params">
                                <span>Trees: {modelComparison[target].random_forest.parameters?.n_estimators || 'N/A'}</span>
                                <span>Depth: {modelComparison[target].random_forest.parameters?.max_depth || 'N/A'}</span>
                              </div>
                            </div>
                          )}
                          {modelComparison[target]?.xgboost && (
                            <div className="model-info-card xgb">
                              <div className="model-badge">⚡ XGBoost</div>
                              <div className="model-params">
                                <span>Trees: {modelComparison[target].xgboost.parameters?.n_estimators || 'N/A'}</span>
                                <span>LR: {modelComparison[target].xgboost.parameters?.learning_rate || 'N/A'}</span>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>

                  {modelComparison.summary?.recommendations && (
                    <div className="recommendations-section">
                      <h4>💡 Recommendations</h4>
                      {modelComparison.summary.recommendations.map((rec, idx) => (
                        <div key={idx} className="recommendation-card">
                          <strong>{rec.scenario}</strong>
                          <p>{rec.recommendation}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <p className="no-data">Model comparison data not available</p>
              )}
            </Card>
          </div>
        )}

        {activeTab === 'seasonal' && (
          <div className="seasonal-section">
            <Card className="chart-card full-width">
              <h3>🗓️ Seasonal Tourism Patterns</h3>
              {seasonalPatterns?.monthly ? (
                <>
                  <ResponsiveContainer width="100%" height={350}>
                    <BarChart data={seasonalPatterns.monthly}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="monthName" />
                      <YAxis />
                      <Tooltip formatter={(value) => value?.toLocaleString()} />
                      <Legend />
                      <Bar dataKey="avgArrivals" name="Avg Arrivals" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>

                  <div className="seasonal-insights">
                    <div className="insight-box peak">
                      <h4>🔥 Peak Season</h4>
                      <p>{seasonalPatterns.peakSeason?.join(', ')}</p>
                    </div>
                    <div className="insight-box low">
                      <h4>📉 Low Season</h4>
                      <p>{seasonalPatterns.lowSeason?.join(', ')}</p>
                    </div>
                  </div>

                  {seasonalPatterns.insights && (
                    <div className="insights-list">
                      <h4>📊 Key Insights</h4>
                      <ul>
                        {seasonalPatterns.insights.map((insight, idx) => (
                          <li key={idx}>{insight}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </>
              ) : (
                <p className="no-data">Seasonal data not available. Run the historical data seeder first.</p>
              )}
            </Card>
          </div>
        )}

        {activeTab === 'forecast' && (
          <div className="forecast-section">
            <Card className="chart-card full-width">
              <div className="forecast-header">
                <h3>🔮 12-Month Tourism Forecast</h3>
                <div className="model-selector">
                  <button 
                    className={`model-btn ${selectedModel === 'rf' ? 'active' : ''}`}
                    onClick={() => loadForecast('rf')}
                  >
                    🌲 Random Forest
                  </button>
                  <button 
                    className={`model-btn ${selectedModel === 'xgb' ? 'active' : ''}`}
                    onClick={() => loadForecast('xgb')}
                  >
                    ⚡ XGBoost
                  </button>
                </div>
              </div>

              {forecast && forecast.length > 0 ? (
                <>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={forecast}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="monthName" />
                      <YAxis />
                      <Tooltip 
                        formatter={(value) => typeof value === 'number' ? value.toLocaleString() : value}
                      />
                      <Legend />
                      <Line 
                        type="monotone" 
                        dataKey="predictions.arrivals" 
                        name="Predicted Arrivals"
                        stroke="#3b82f6" 
                        strokeWidth={3}
                        dot={{ fill: '#3b82f6', strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>

                  <div className="forecast-table">
                    <h4>Forecast Details</h4>
                    <table>
                      <thead>
                        <tr>
                          <th>Month</th>
                          <th>Arrivals</th>
                          <th>Revenue (LKR M)</th>
                          <th>Occupancy (%)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {forecast.map((f, idx) => (
                          <tr key={idx}>
                            <td>{f.monthName} {f.year}</td>
                            <td>{f.predictions?.arrivals?.toLocaleString() || '-'}</td>
                            <td>{f.predictions?.revenue ? `LKR ${(f.predictions.revenue / 1000000).toFixed(2)}M` : '-'}</td>
                            <td>{f.predictions?.occupancy ? `${(f.predictions.occupancy * 100).toFixed(1)}%` : '-'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <p className="no-data">Forecast data not available</p>
              )}
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalyticsPage;
