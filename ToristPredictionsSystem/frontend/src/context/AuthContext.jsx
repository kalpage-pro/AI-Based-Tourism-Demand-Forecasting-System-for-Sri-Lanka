import { createContext, useState, useEffect } from 'react';
import authService from '../services/auth.service';

export const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      console.log('🔐 Initializing authentication...');
      const token = authService.getToken();
      const storedUser = authService.getUser();
      
      console.log('Token exists:', !!token);
      console.log('Stored user:', storedUser);
      
      if (token && storedUser) {
        try {
          console.log('Validating token...');
          const userData = await authService.getCurrentUser();
          console.log('✅ Token valid, user:', userData.data);
          setUser(userData.data);
        } catch (error) {
          console.error('❌ Token validation failed:', error);
          authService.logout();
          setUser(null);
        }
      } else {
        console.log('No token or stored user found');
      }
      
      setLoading(false);
    };

    initAuth();
  }, []);

  const login = async (credentials) => {
    try {
      console.log('🔑 Attempting login...');
      const data = await authService.login(credentials);
      console.log('✅ Login successful:', data.user);
      setUser(data.user);
      return data;
    } catch (error) {
      console.error('❌ Login failed:', error);
      throw error;
    }
  };

  const register = async (userData) => {
    try {
      console.log('📝 Attempting registration...');
      const data = await authService.register(userData);
      console.log('✅ Registration successful:', data.user);
      setUser(data.user);
      return data;
    } catch (error) {
      console.error('❌ Registration failed:', error);
      throw error;
    }
  };

  const logout = () => {
    console.log('👋 Logging out...');
    authService.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}