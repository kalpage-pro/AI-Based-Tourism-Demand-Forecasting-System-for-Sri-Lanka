import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from '../../hooks/useAuth'
import './Auth.css'

function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const { login } = useAuth();
  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();
    setErr("");
    setLoading(true);

    try {
      const response = await login({ email, password });
      console.log('🔑 Login response:', response);
      console.log('👤 User role:', response.user?.role);
      
      setLoading(false);
      // Redirect admin users to admin panel, others to home
      if (response.user?.role === 'admin') {
        console.log('🔐 Redirecting admin to /admin');
        navigate('/admin', { replace: true });
      } else {
        console.log('🏠 Redirecting user to /');
        navigate('/', { replace: true });
      }
    } catch (error) {
      setLoading(false);
      setErr(error.response?.data?.message || error.message || 'Login failed');
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <div className="left-panel">
          <h2>WELCOME TO</h2>
          <div className="logo">Tourist Predictions</div>
          <p>Your intelligent solution for tourism forecasting in Sri Lanka.</p>
        </div>

        <div className="right-panel">
          <h3>Login to Dashboard</h3>

          <div className="social-row">
            <button type="button" className="social facebook">Continue with Facebook</button>
            <button type="button" className="social google">Continue with Google</button>
          </div>

          <div className="or-divider">OR</div>

          <form className="auth-form" onSubmit={submit}>
            <input
              required
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <input
              required
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />

            <div className="form-row">
              <a className="forgot" href="#">Forgot Password?</a>
            </div>

            {err && <div className="error">{err}</div>}

            <button className="login-btn" type="submit" disabled={loading}>
              {loading ? "Logging in..." : "LOGIN"}
            </button>

            <div className="signup-line">
              If you are a new user, <Link to="/register">Signup here</Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Login;