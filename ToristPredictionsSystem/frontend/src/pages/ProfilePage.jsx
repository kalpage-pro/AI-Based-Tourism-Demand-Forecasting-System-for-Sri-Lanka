import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import Card from '../components/Common/Card'
import Input from '../components/Common/Input'
import Button from '../components/Common/Button'
import './Pages.css'

function ProfilePage() {
  const { user, updateProfile } = useAuth()
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  })
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setMessage('')
    setLoading(true)

    try {
      await updateProfile(formData)
      setMessage('Profile updated successfully!')
      setFormData({
        ...formData,
        currentPassword: '',
        newPassword: '',
        confirmPassword: ''
      })
    } catch (err) {
      setMessage(err.response?.data?.message || 'Update failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page profile-page">
      <div className="page-header">
        <h1 className="page-title">Profile Settings</h1>
        <p className="page-subtitle">Manage your account information</p>
      </div>

      <div className="profile-content">
        <Card title="Account Information" icon="👤">
          <form className="profile-form" onSubmit={handleSubmit}>
            {message && (
              <div className={`message ${message.includes('success') ? 'success' : 'error'}`}>
                {message}
              </div>
            )}

            <Input
              label="Full Name"
              name="name"
              type="text"
              value={formData.name}
              onChange={handleChange}
              icon="👤"
            />

            <Input
              label="Email Address"
              name="email"
              type="email"
              value={formData.email}
              onChange={handleChange}
              icon="✉️"
              disabled
            />

            <div className="form-divider">
              <span>Change Password</span>
            </div>

            <Input
              label="Current Password"
              name="currentPassword"
              type="password"
              value={formData.currentPassword}
              onChange={handleChange}
              icon="🔒"
            />

            <Input
              label="New Password"
              name="newPassword"
              type="password"
              value={formData.newPassword}
              onChange={handleChange}
              icon="🔑"
            />

            <Input
              label="Confirm New Password"
              name="confirmPassword"
              type="password"
              value={formData.confirmPassword}
              onChange={handleChange}
              icon="🔑"
            />

            <Button type="submit" loading={loading} fullWidth>
              Update Profile
            </Button>
          </form>
        </Card>

        <Card title="Account Stats" icon="📊">
          <div className="profile-stats">
            <div className="stat-item">
              <span className="stat-icon">📈</span>
              <div>
                <span className="stat-label">Total Predictions</span>
                <span className="stat-value">{user?.totalPredictions || 0}</span>
              </div>
            </div>
            <div className="stat-item">
              <span className="stat-icon">📅</span>
              <div>
                <span className="stat-label">Member Since</span>
                <span className="stat-value">
                  {user?.createdAt 
                    ? new Date(user.createdAt).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric'
                      })
                    : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

export default ProfilePage