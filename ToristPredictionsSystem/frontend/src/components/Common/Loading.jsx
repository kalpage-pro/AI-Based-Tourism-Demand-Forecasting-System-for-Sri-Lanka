import './Common.css'

function Loading({ message = 'Loading...', size = 'medium' }) {
  return (
    <div className="loading-container">
      <div className={`loading-spinner loading-${size}`}>
        <div className="spinner-circle"></div>
      </div>
      {message && <p className="loading-message">{message}</p>}
    </div>
  )
}

export default Loading