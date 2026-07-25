import React, { createContext, useState, useContext, useEffect } from 'react';
import { documentAPI } from '../api/api'; 

const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [username, setUsername] = useState(localStorage.getItem('username') || '');

  const login = (token, userId, username) => {
    localStorage.setItem('token', token);
    const uniqueSessionId = `${userId}_${Date.now()}`;
    localStorage.setItem('session_id', uniqueSessionId);
    localStorage.setItem('username', username); 
    setUsername(username); 
    setIsAuthenticated(true);
  };

  const logout = async () => {
    try {
      const sessionId = localStorage.getItem('session_id');
      if (sessionId) {
        await documentAPI.cleanupData(sessionId);
      }
    } catch (error) {
      console.error("Backend cleanup failed, but proceeding with local logout", error);
    } finally {
      localStorage.removeItem('token');
      localStorage.removeItem('session_id');
      localStorage.removeItem('username'); 
      setUsername(''); 
      setIsAuthenticated(false);
    }
  };

  const toggleDarkMode = () => setIsDarkMode(!isDarkMode);

  return (
    <AppContext.Provider value={{ isAuthenticated, isDarkMode, username, login, logout, toggleDarkMode }}>
      <div className={`${isDarkMode ? 'dark' : ''} font-sans h-screen`}>
        {children}
      </div>
    </AppContext.Provider>
  );
};

export const useAppContext = () => useContext(AppContext);