// components/NavBar.jsx
import React, { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContext } from './Authcontext';
import { FaUserCircle } from 'react-icons/fa';

const NavBar = () => {
  const { user, logout } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleLoginClick = () => {
    navigate('/auth', { state: { mode: 'login' } });
  };

  const handleSignupClick = () => {
    navigate('/auth', { state: { mode: 'signup' } });
  };

  const handleLogoutClick = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="absolute top-4 right-6 flex items-center gap-4 z-10">
      {!user ? (
        <>
          <button
            onClick={handleLoginClick}
            className="bg-white text-blue-600 font-semibold px-4 py-2 rounded-full hover:bg-gray-100 shadow-md transition duration-200"
          >
            Login
          </button>
          <button
            onClick={handleSignupClick}
            className="bg-blue-600 text-white font-semibold px-4 py-2 rounded-full hover:bg-blue-700 shadow-md transition duration-200"
          >
            Signup
          </button>
        </>
      ) : (
        <div className="flex items-center gap-2">
          <FaUserCircle size={28} className="text-cyan-400 hover:text-cyan-600 cursor-pointer" onClick={()=>navigate("/profile")}/>
          <button
            onClick={handleLogoutClick}
            className="bg-red-500 text-white font-semibold px-4 py-2 rounded-full hover:bg-red-600 shadow-md transition duration-200"
          >
            Logout
          </button>
        </div>
      )}
    </div>
  );
};

export default NavBar;
