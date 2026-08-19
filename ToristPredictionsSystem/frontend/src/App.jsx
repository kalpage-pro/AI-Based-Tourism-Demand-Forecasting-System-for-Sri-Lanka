import { Routes, Route, Navigate } from 'react-router-dom'
import { useState } from 'react'
import { useAuth } from './hooks/useAuth'
import Navbar from './components/Layout/Navbar'
import Sidebar from './components/Layout/Sidebar'
import Footer from './components/Layout/Footer'
import Home from './pages/Home'
import Login from './components/Auth/Login'
import Register from './components/Auth/Register'
import PredictionPage from './pages/PredictionPage'
import HistoryPage from './pages/HistoryPage'
import ProfilePage from './pages/ProfilePage'
import AdminPage from './pages/AdminPage'
import DownloadsPage from './pages/DownloadsPage'
import AnalyticsPage from './pages/AnalyticsPage'
import DestinationsPage from './pages/DestinationsPage'
import ScenarioPage from './pages/ScenarioPage'
import ReportsPage from './pages/ReportsPage'
import Loading from './components/Common/Loading'
import './App.css'

function App() {
  const { user, loading } = useAuth()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev)
  }

  if (loading) {
    return <Loading message="Loading application..." />
  }

  return (
    <div className="app">
      {user && <Navbar onToggleSidebar={toggleSidebar} sidebarOpen={sidebarOpen} />}
      <div className={`app-container ${sidebarOpen ? 'sidebar-open' : ''}`}>
        {user && <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />}
        <main className="main-content">
          <Routes>
            <Route path="/login" element={!user ? <Login /> : <Navigate to={user.role === 'admin' ? '/admin' : '/'} />} />
            <Route path="/register" element={!user ? <Register /> : <Navigate to={user.role === 'admin' ? '/admin' : '/'} />} />
            <Route path="/" element={user ? (user.role === 'admin' ? <Navigate to="/admin" /> : <Home />) : <Navigate to="/login" />} />
            <Route path="/predict" element={user ? <PredictionPage /> : <Navigate to="/login" />} />
            <Route path="/history" element={user ? <HistoryPage /> : <Navigate to="/login" />} />
            <Route path="/profile" element={user ? <ProfilePage /> : <Navigate to="/login" />} />
            <Route path="/downloads" element={user ? <DownloadsPage /> : <Navigate to="/login" />} />
            <Route path="/analytics" element={user ? <AnalyticsPage /> : <Navigate to="/login" />} />
            <Route path="/destinations" element={user ? <DestinationsPage /> : <Navigate to="/login" />} />
            <Route path="/scenarios" element={user ? <ScenarioPage /> : <Navigate to="/login" />} />
            <Route path="/reports" element={user ? <ReportsPage /> : <Navigate to="/login" />} />
            <Route 
              path="/admin" 
              element={
                user?.role === 'admin' 
                  ? <AdminPage /> 
                  : <Navigate to="/" />
              }
            />
          </Routes>
        </main>
      </div>
      {user && <Footer />}
    </div>
  )
}

export default App
