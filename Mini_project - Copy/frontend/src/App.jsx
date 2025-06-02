import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import Main from './components/Main'; 
import NavBar from "./components/NavBar";
import AuthPage from "./components/AuthPage";
import UserProfile from "./components/UserProfile";


const App = () => {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/home" element={<Main />} />
        <Route path="/auth" element={<AuthPage />} />
        <Route path="/profile" element={<UserProfile />} />
      </Routes>
    </Router>
  );
};

export default App;
