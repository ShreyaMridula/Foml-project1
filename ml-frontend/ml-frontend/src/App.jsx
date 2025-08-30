import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import HomePage from './pages/HomePage';
import SignUpPage from './pages/SignupPage';
import LoginPage from './pages/LoginPage';
import UploadPage from './pages/UploadPage';

import AccountPage from './pages/AccountPage';
import SearchResultPage from './pages/SearchResultPage';
import WatchPage from "./pages/WatchPage";
import TrendingPage from './pages/TrendingPage';
import SubscriptionPage from "./pages/SubscriptionPage";
import LibraryPage from "./pages/LibraryPage";

function App() {
  return (
    <div>
      <Routes>
        {/* Default route: redirect to login */}
        <Route path="/" element={<Navigate to="/login" />} />

        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignUpPage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/watch/:id" element={<WatchPage />} /> {/* Corrected this line */}
        <Route path="/account" element={<AccountPage />} />
        <Route path="/search" element={<SearchResultPage />} />
        <Route path="/trending" element={<TrendingPage />} />
        <Route path="/subscriptions" element={<SubscriptionPage />} />
        <Route path="/library" element={<LibraryPage />} />
        

      </Routes>
    </div>
  );
}

export default App;
