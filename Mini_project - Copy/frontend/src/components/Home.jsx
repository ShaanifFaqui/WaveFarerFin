import React, { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import homeBg from '../assets/home.avif';
import NavBar from '../components/NavBar';
import { AuthContext } from '../components/Authcontext'; 

const Home = () => {
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleExploreClick = () => {
    if (user) {
      navigate('/home');
    } else {
      alert('Please login to explore!');
    }
  };

  return (
    <div
      className="relative h-screen w-full bg-cover bg-center"
      style={{ backgroundImage: `url(${homeBg})` }}
    >
      <NavBar />

      <div className="absolute inset-0 bg-black bg-opacity-60 flex flex-col justify-center items-center text-white text-center px-4">
        <h1 className="text-4xl md:text-6xl font-bold mb-4">🌊 WaveFarer</h1>
        <p className="text-lg md:text-2xl max-w-xl mb-8">
          Get real-time alerts, sea conditions, and stay safe on your next coastal adventure.
        </p>
        <button
          onClick={handleExploreClick}
          className={`px-6 py-3 rounded-full text-lg font-medium shadow-lg transition-transform duration-300 transform hover:scale-105 ${
            user ? 'bg-blue-500 hover:bg-blue-600' : 'bg-gray-400 cursor-not-allowed'
          }`}
        >
          Explore Now
        </button>
      </div>
    </div>
  );
};

export default Home;
