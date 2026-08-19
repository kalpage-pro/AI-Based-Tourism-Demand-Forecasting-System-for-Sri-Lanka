import { NavLink } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'
import './Layout.css'

function Sidebar({ isOpen, onClose }) {
  const { user } = useAuth()
  
  const menuItems = [
    { path: '/', icon: '📊', label: 'Dashboard' },
    { path: '/predict', icon: '🎯', label: 'New Prediction' },
    { path: '/destinations', icon: '🏝️', label: 'Destinations' },
    { path: '/analytics', icon: '📈', label: 'Analytics' },
    { path: '/scenarios', icon: '🎮', label: 'Scenarios' },
    { path: '/history', icon: '📜', label: 'History' },
    { path: '/reports', icon: '📑', label: 'Reports' },
    { path: '/downloads', icon: '📥', label: 'Downloads' },
    { path: '/profile', icon: '⚙️', label: 'Settings' },
  ]

  // Add admin panel link for admin users
  if (user?.role === 'admin') {
    menuItems.push({ path: '/admin', icon: '👑', label: 'Admin Panel' })
  }

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div className="sidebar-overlay" onClick={onClose}></div>
      )}
      
      {/* Sidebar */}
      <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
        {/* Close button for mobile */}
        <button className="sidebar-close" onClick={onClose}>
          <span>✕</span>
        </button>

        <nav className="sidebar-nav">
          {menuItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
              onClick={onClose}
            >
              <span className="sidebar-icon">{item.icon}</span>
              <span className="sidebar-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  )
}

export default Sidebar