import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from '../../hooks/useAuth'
import './Auth.css'

function Register() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const { register } = useAuth();
  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();
    setErr("");

    if (password.length < 6) {
      setErr('Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    try {
      await register({ name, email, password });
      setLoading(false);
      navigate("/login");
    } catch (error) {
      setLoading(false);
      setErr(error.response?.data?.message || error.message || 'Signup failed');
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-card small">
        <div className="left-panel sign-left">
          <h2>Create Account</h2>
          <p>Join Tourist Predictions — it's quick and free.</p>
        </div>

        <div className="right-panel">
          <form className="auth-form" onSubmit={submit}>
            <input 
              required 
              placeholder="Full name" 
              value={name} 
              onChange={(e) => setName(e.target.value)} 
            />
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
              placeholder="Password (min 6 characters)" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
            />

            {err && <div className="error">{err}</div>}

            <button className="login-btn" type="submit" disabled={loading}>
              {loading ? "Signing up..." : "SIGN UP"}
            </button>

            <div className="signup-line">
              Already a user? <Link to="/login">Login</Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Register;