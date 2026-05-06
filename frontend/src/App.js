import React, { useState } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import Auth from './components/Auth';
import MLOpsAgent from './components/MLOpsAgent';

// Main app content with routing logic
function AppContent() {
  const { user, login, logout, isLoading } = useAuth();
  const [currentPage, setCurrentPage] = useState('dashboard'); // 'auth', 'dashboard'
  const [authMode, setAuthMode] = useState('login');

  // Show loading spinner while checking auth
  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // If user is logged in, show dashboard
  if (user) {
    return (
      <div className="relative">
        {/* User menu overlay */}
        <div className="fixed top-4 right-4 z-50 flex items-center gap-3">
          <div className="px-4 py-2 bg-gray-800/80 backdrop-blur rounded-lg border border-white/10 flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-orange-500 to-yellow-500 flex items-center justify-center text-sm font-bold">
              {user.name?.charAt(0).toUpperCase() || 'U'}
            </div>
            <span className="text-sm text-gray-300">{user.name}</span>
            <button
              onClick={logout}
              className="text-sm text-gray-500 hover:text-white transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
        <MLOpsAgent />
      </div>
    );
  }

  // Show auth page
  if (currentPage === 'auth') {
    return (
      <Auth
        initialMode={authMode}
        onLogin={(userData) => {
          login(userData);
          setCurrentPage('dashboard');
        }}
        onBack={() => setCurrentPage('dashboard')}
      />
    );
  }

  return (
    <div className="relative">
      <div className="fixed right-3 top-3 z-50 flex max-w-[calc(100vw-1.5rem)] items-center gap-2 rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-2 text-xs text-slate-300 shadow-lg shadow-black/20 backdrop-blur">
        {user ? (
          <>
            <span className="min-w-0 truncate">{user.name}</span>
            <button onClick={logout} className="rounded-md border border-slate-700 px-2 py-1 hover:bg-slate-900">
              Logout
            </button>
          </>
        ) : (
          <>
            <span className="hidden text-slate-500 sm:inline">Demo console</span>
            <button
              onClick={() => {
                setAuthMode('login');
                setCurrentPage('auth');
              }}
              className="rounded-md border border-slate-700 px-2 py-1 hover:bg-slate-900"
            >
              Sign in
            </button>
            <button
              onClick={() => {
                setAuthMode('signup');
                setCurrentPage('auth');
              }}
              className="rounded-md bg-cyan-600 px-2 py-1 font-semibold text-white hover:bg-cyan-500"
            >
              Create account
            </button>
          </>
        )}
      </div>
      <MLOpsAgent />
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
