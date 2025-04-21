import { createContext, useContext, useState, useEffect } from 'react';
import { useAuth0 } from '@auth0/auth0-react';

// Authentication context
const AuthContext = createContext();

// Custom hook to use the auth context
export const useAuth = () => useContext(AuthContext);

// Auth provider component
export const AuthProvider = ({ children }) => {
  // Auth0 hooks
  const { 
    isAuthenticated, 
    isLoading, 
    user, 
    loginWithRedirect, 
    logout: auth0Logout,
    getAccessTokenSilently
  } = useAuth0();

  // Get user profile
  const getUserProfile = async () => {
    return user;
  };

  // Get token
  const getToken = async () => {
    return await getAccessTokenSilently();
  };

  // u00d6zelleu015ftirilmiu015f login fonksiyonu
  const loginWithAuth0 = (options = {}) => {
    // Auth0'nun varsayu0131lan 'Log in to dev-...' mesaju0131 yerine
    // dogrudan giriş işlemini bau015flatma
    return loginWithRedirect({
      ...options,
      authorizationParams: {
        ...options.authorizationParams,
        ui_locales: 'tr',
        login_hint: 'GITGUD Afet Hasar Tespit Sistemi',
        screen_hint: 'login',
        prompt: 'login'
      },
      appState: {
        returnTo: window.location.origin,
        ...options.appState
      }
    });
  };

  // Context value
  const value = {
    isAuthenticated,
    isLoading,
    user,
    authType: 'auth0',
    loginWithAuth0,  // u00d6zelleu015ftirilmiu015f fonksiyonu kullan
    logout: auth0Logout,
    getToken,
    getUserProfile
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;
