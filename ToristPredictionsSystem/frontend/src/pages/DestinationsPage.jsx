import { useState, useEffect } from 'react';
import {
  getAllDestinations,
  getFeaturedDestinations,
  getEconomicalFlights,
  getBestHotels
} from '../services/destination.service';
import Loading from '../components/Common/Loading';
import Card from '../components/Common/Card';
import './DestinationsPage.css';

function DestinationsPage() {
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('all');
  const [destinations, setDestinations] = useState([]);
  const [featuredDestinations, setFeaturedDestinations] = useState([]);
  const [economicalFlights, setEconomicalFlights] = useState([]);
  const [bestHotels, setBestHotels] = useState([]);
  const [error, setError] = useState(null);
  
  // Filters
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [regionFilter, setRegionFilter] = useState('all');
  const [hotelFilter, setHotelFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  const categories = ['all', 'beach', 'cultural', 'wildlife', 'adventure', 'hill-country', 'historical', 'religious', 'nature'];
  const regions = ['all', 'North', 'South', 'East', 'West', 'Central', 'North Central', 'North Western', 'Sabaragamuwa', 'Uva'];

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [allRes, featRes, flightsRes, hotelsRes] = await Promise.all([
        getAllDestinations({ limit: 100 }).catch(() => ({ data: [] })),
        getFeaturedDestinations().catch(() => ({ data: [] })),
        getEconomicalFlights().catch(() => ({ data: [] })),
        getBestHotels().catch(() => ({ data: [] }))
      ]);

      setDestinations(allRes.data || []);
      setFeaturedDestinations(featRes.data || []);
      setEconomicalFlights(flightsRes.data || []);
      setBestHotels(hotelsRes.data || []);
    } catch (err) {
      setError('Failed to load destinations');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const filteredDestinations = destinations.filter(dest => {
    const matchesCategory = categoryFilter === 'all' || dest.category === categoryFilter;
    const matchesRegion = regionFilter === 'all' || dest.region === regionFilter;
    const matchesSearch = !searchQuery || 
      dest.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      dest.description?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesRegion && matchesSearch;
  });

  const filteredHotels = bestHotels.filter(hotel => {
    return hotelFilter === 'all' || hotel.priceRange === hotelFilter;
  });

  if (loading) {
    return <Loading message="Loading destinations..." />;
  }

  return (
    <div className="destinations-page">
      <div className="destinations-header">
        <div className="header-content">
          <h1>🌴 Sri Lanka Destinations</h1>
          <p>Explore the best tourist destinations, hotels, and travel options</p>
        </div>
        <div className="header-stats">
          <div className="header-stat">
            <span className="stat-number">{destinations.length}</span>
            <span className="stat-label">Destinations</span>
          </div>
          <div className="header-stat">
            <span className="stat-number">{bestHotels.length}</span>
            <span className="stat-label">Hotels</span>
          </div>
          <div className="header-stat">
            <span className="stat-number">{economicalFlights.length}</span>
            <span className="stat-label">Routes</span>
          </div>
        </div>
      </div>

      <div className="destinations-tabs">
        <button 
          className={`dest-tab ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          🗺️ All Destinations
        </button>
        <button 
          className={`dest-tab ${activeTab === 'featured' ? 'active' : ''}`}
          onClick={() => setActiveTab('featured')}
        >
          ⭐ Featured
        </button>
        <button 
          className={`dest-tab ${activeTab === 'hotels' ? 'active' : ''}`}
          onClick={() => setActiveTab('hotels')}
        >
          🏨 Hotels
        </button>
        <button 
          className={`dest-tab ${activeTab === 'travel' ? 'active' : ''}`}
          onClick={() => setActiveTab('travel')}
        >
          ✈️ Travel Options
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="destinations-content">
        {/* All Destinations Tab */}
        {activeTab === 'all' && (
          <div className="all-destinations">
            <div className="filters-bar">
              <div className="search-box">
                <span className="search-icon">🔍</span>
                <input
                  type="text"
                  placeholder="Search destinations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
              <select 
                value={categoryFilter} 
                onChange={(e) => setCategoryFilter(e.target.value)}
              >
                {categories.map(cat => (
                  <option key={cat} value={cat}>
                    {cat === 'all' ? 'All Categories' : cat.charAt(0).toUpperCase() + cat.slice(1).replace('-', ' ')}
                  </option>
                ))}
              </select>
              <select 
                value={regionFilter} 
                onChange={(e) => setRegionFilter(e.target.value)}
              >
                {regions.map(reg => (
                  <option key={reg} value={reg}>
                    {reg === 'all' ? 'All Regions' : reg}
                  </option>
                ))}
              </select>
            </div>

            <div className="results-count">
              Showing {filteredDestinations.length} destination{filteredDestinations.length !== 1 ? 's' : ''}
            </div>

            {filteredDestinations.length === 0 ? (
              <div className="no-results">
                <span>🔍</span>
                <p>No destinations found matching your criteria</p>
              </div>
            ) : (
              <div className="destinations-grid">
                {filteredDestinations.map((dest) => (
                  <div key={dest._id} className="destination-card">
                    <div className="dest-card-top">
                      <div className="dest-badges">
                        <span className={`category-badge ${dest.category}`}>
                          {dest.category?.replace('-', ' ')}
                        </span>
                        {dest.isFeatured && <span className="featured-tag">⭐ Featured</span>}
                      </div>
                      <h3>{dest.name}</h3>
                      <p className="dest-region">📍 {dest.region}</p>
                    </div>
                    <div className="dest-card-body">
                      <p className="dest-description">{dest.description?.substring(0, 120)}...</p>
                      
                      {dest.highlights && dest.highlights.length > 0 && (
                        <div className="dest-highlights">
                          {dest.highlights.slice(0, 3).map((h, i) => (
                            <span key={i} className="highlight-chip">{h}</span>
                          ))}
                        </div>
                      )}

                      <div className="dest-meta">
                        <div className="meta-item">
                          <span className="meta-icon">👥</span>
                          <span>{dest.yearlyArrivals?.toLocaleString()}</span>
                          <span className="meta-label">visitors/year</span>
                        </div>
                        <div className="meta-item">
                          <span className="meta-icon">⭐</span>
                          <span>{dest.ratings?.overall?.toFixed(1) || 'N/A'}</span>
                          <span className="meta-label">rating</span>
                        </div>
                        <div className="meta-item">
                          <span className="meta-icon">📅</span>
                          <span>{dest.averageStayDays}</span>
                          <span className="meta-label">days avg</span>
                        </div>
                      </div>

                      {dest.bestTimeToVisit && (
                        <div className="best-time">
                          <span>🗓️ Best time: {dest.bestTimeToVisit}</span>
                        </div>
                      )}

                      {/* Quick info about hotels and routes */}
                      <div className="dest-quick-info">
                        <span>🏨 {dest.hotels?.length || 0} hotels</span>
                        <span>✈️ {dest.flights?.length || 0} routes</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Featured Tab */}
        {activeTab === 'featured' && (
          <div className="featured-destinations">
            <div className="section-intro">
              <h2>⭐ Top Featured Destinations</h2>
              <p>Handpicked must-visit locations in Sri Lanka</p>
            </div>

            {featuredDestinations.length === 0 ? (
              <div className="no-results">
                <span>⭐</span>
                <p>No featured destinations available</p>
              </div>
            ) : (
              <div className="featured-grid">
                {featuredDestinations.map((dest, idx) => (
                  <Card key={dest._id} className={`featured-card rank-${idx + 1}`}>
                    <div className="featured-rank">#{idx + 1}</div>
                    <div className="featured-content">
                      <h3>{dest.name}</h3>
                      <span className={`category-badge ${dest.category}`}>{dest.category}</span>
                      <p className="region">📍 {dest.region}</p>
                      <p className="description">{dest.description?.substring(0, 150)}...</p>
                      
                      <div className="featured-stats">
                        <div className="f-stat">
                          <strong>{dest.yearlyArrivals?.toLocaleString()}</strong>
                          <span>arrivals/year</span>
                        </div>
                        <div className="f-stat">
                          <strong>{dest.popularity}%</strong>
                          <span>popularity</span>
                        </div>
                        <div className="f-stat">
                          <strong>{dest.ratings?.overall?.toFixed(1) || '-'}</strong>
                          <span>rating</span>
                        </div>
                      </div>

                      {dest.highlights && (
                        <div className="featured-highlights">
                          {dest.highlights.slice(0, 4).map((h, i) => (
                            <span key={i}>{h}</span>
                          ))}
                        </div>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Hotels Tab */}
        {activeTab === 'hotels' && (
          <div className="hotels-section">
            <div className="section-intro">
              <h2>🏨 Best Hotels & Accommodations</h2>
              <p>Find the perfect place to stay during your visit</p>
            </div>

            <div className="hotel-filters">
              <button 
                className={`filter-btn ${hotelFilter === 'all' ? 'active' : ''}`}
                onClick={() => setHotelFilter('all')}
              >
                All Hotels
              </button>
              <button 
                className={`filter-btn ${hotelFilter === 'luxury' ? 'active' : ''}`}
                onClick={() => setHotelFilter('luxury')}
              >
                💎 Luxury
              </button>
              <button 
                className={`filter-btn ${hotelFilter === 'mid-range' ? 'active' : ''}`}
                onClick={() => setHotelFilter('mid-range')}
              >
                🏠 Mid-Range
              </button>
              <button 
                className={`filter-btn ${hotelFilter === 'budget' ? 'active' : ''}`}
                onClick={() => setHotelFilter('budget')}
              >
                💰 Budget
              </button>
            </div>

            {filteredHotels.length === 0 ? (
              <div className="no-results">
                <span>🏨</span>
                <p>No hotels found</p>
              </div>
            ) : (
              <div className="hotels-grid">
                {filteredHotels.map((hotel, idx) => (
                  <div key={idx} className={`hotel-card ${hotel.priceRange}`}>
                    <div className="hotel-header">
                      <h4>{hotel.name}</h4>
                      <div className="hotel-stars">
                        {'⭐'.repeat(hotel.rating || 0)}
                      </div>
                    </div>
                    <p className="hotel-location">
                      📍 {hotel.destination}, {hotel.region}
                    </p>
                    <div className="hotel-pricing">
                      <span className={`price-badge ${hotel.priceRange}`}>
                        {hotel.priceRange?.replace('-', ' ')}
                      </span>
                      {hotel.pricePerNight && (
                        <span className="price-amount">${hotel.pricePerNight}/night</span>
                      )}
                    </div>
                    {hotel.amenities && hotel.amenities.length > 0 && (
                      <div className="hotel-amenities">
                        {hotel.amenities.slice(0, 5).map((amenity, i) => (
                          <span key={i} className="amenity">{amenity}</span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Travel Options Tab */}
        {activeTab === 'travel' && (
          <div className="travel-section">
            <div className="section-intro">
              <h2>✈️ Travel & Transportation</h2>
              <p>Economical ways to explore Sri Lanka</p>
            </div>

            <div className="travel-tips-grid">
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
                <h4>🗓️ Best Time to Visit</h4>
                <ul>
                  <li><strong>December - March:</strong> Peak season, ideal weather on west coast</li>
                  <li><strong>April - September:</strong> Best for east coast beaches</li>
                  <li><strong>Year-round:</strong> Hill country and cultural triangle</li>
                </ul>
              </Card>
            </div>

            <h3 className="routes-title">🚌 Economical Routes</h3>
            
            {economicalFlights.length === 0 ? (
              <div className="no-results">
                <span>🚌</span>
                <p>No travel routes found</p>
              </div>
            ) : (
              <div className="routes-table-wrapper">
                <table className="routes-table">
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
                    {economicalFlights.map((route, idx) => (
                      <tr key={idx}>
                        <td className="transport-cell">
                          <span className="transport-icon">
                            {route.airline?.toLowerCase().includes('train') ? '🚂' : 
                             route.airline?.toLowerCase().includes('bus') ? '🚌' : '✈️'}
                          </span>
                          {route.airline}
                        </td>
                        <td>{route.from}</td>
                        <td>{route.destination}</td>
                        <td>{route.duration}</td>
                        <td>{route.frequency}</td>
                        <td className="price-cell">
                          <span className="route-price">${route.price}</span>
                          {route.isEconomical && <span className="eco-tag">💰 Budget</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default DestinationsPage;
