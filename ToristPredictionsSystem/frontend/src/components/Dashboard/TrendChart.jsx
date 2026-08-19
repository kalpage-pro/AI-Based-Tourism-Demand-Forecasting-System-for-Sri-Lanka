import './Dashboard.css'

function TrendChart({ data, title }) {
  if (!data || data.length === 0) {
    return (
      <div className="trend-chart">
        <p className="no-data">No trend data available</p>
      </div>
    )
  }

  const maxValue = Math.max(...data.map(d => d.value))

  return (
    <div className="trend-chart">
      <h3 className="chart-title">{title}</h3>
      <div className="chart-bars">
        {data.map((item, index) => (
          <div key={index} className="chart-bar-wrapper">
            <div 
              className="chart-bar"
              style={{ height: `${(item.value / maxValue) * 100}%` }}
              title={`${item.label}: ${item.value}`}
            >
              <span className="bar-value">{item.value}</span>
            </div>
            <span className="bar-label">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default TrendChart