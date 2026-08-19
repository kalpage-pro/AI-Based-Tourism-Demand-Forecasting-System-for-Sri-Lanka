import './Common.css'

function Input({ 
  label, 
  error, 
  icon,
  autoComplete,
  ...props 
}) {
  return (
    <div className="input-group">
      {label && <label className="input-label">{label}</label>}
      <div className="input-wrapper">
        {icon && <span className="input-icon">{icon}</span>}
        <input 
          className={`input ${error ? 'input-error' : ''} ${icon ? 'input-with-icon' : ''}`}
          autoComplete={autoComplete}
          {...props}
        />
      </div>
      {error && <span className="input-error-text">{error}</span>}
    </div>
  )
}

export default Input