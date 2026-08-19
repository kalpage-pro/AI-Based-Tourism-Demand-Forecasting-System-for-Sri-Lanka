import './Common.css'

function Card({ 
  children, 
  title, 
  icon, 
  glow = false, 
  className = '', 
  ...props 
}) {
  return (
    <div className={`card ${glow ? 'card-glow' : ''} ${className}`} {...props}>
      {(title || icon) && (
        <div className="card-header">
          {icon && <span className="card-icon">{icon}</span>}
          {title && <h3 className="card-title">{title}</h3>}
        </div>
      )}
      <div className="card-body">{children}</div>
    </div>
  )
}

export default Card