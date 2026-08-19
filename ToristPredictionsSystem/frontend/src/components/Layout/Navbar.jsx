import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'
import './Layout.css'

function Navbar({ onToggleSidebar, sidebarOpen }) {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Left section */}
        <div className="navbar-left">
          {/* Hamburger Menu Button */}
          <button 
            className={`hamburger-btn ${sidebarOpen ? 'open' : ''}`}
            onClick={onToggleSidebar}
            aria-label="Toggle menu"
          >
            <span className="hamburger-line"></span>
            <span className="hamburger-line"></span>
            <span className="hamburger-line"></span>
          </button>

          <Link to="/" className="navbar-brand">
            <span className="brand-icon">🌍</span>
            <span className="brand-text">TouristPredict</span>
          </Link>
        </div>

        {/* Right section */}
        <div className="navbar-right">
          {user && (
            <>
              <div className="user-info">
                <div className="user-avatar">
                  {user.name?.[0]?.toUpperCase() || 'U'}
                </div>
                <span className="user-name">{user.name}</span>
              </div>
              <button onClick={handleLogout} className="logout-btn">
                <span className="logout-icon">🚪</span>
                <span>Logout</span>
              </button>
            </>
          )}
        </div>
      </div>
    </nav>
  )
}

export default Navbar