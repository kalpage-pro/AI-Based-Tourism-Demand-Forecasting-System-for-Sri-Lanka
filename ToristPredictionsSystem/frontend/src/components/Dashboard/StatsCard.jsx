import './Dashboard.css'

function StatsCard({ title, value, icon, color = 'yellow', trend }) {
  return (
    <div className={`stats-card card-${color}`}>
      <div className="card-icon-wrapper">
        <span className="card-icon">{icon}</span>
      </div>
      <div className="card-content">
        <p className="card-title">{title}</p>
        <h3 className="card-value">{value}</h3>
        {trend && (
          <span className={`card-trend trend-${trend.startsWith('+') ? 'up' : trend === 'stable' ? 'stable' : 'down'}`}>
            {trend}
          </span>
        )}
      </div>
    </div>
  )
}

export default StatsCard