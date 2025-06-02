import React, { useEffect, useState, useContext } from 'react';
import {useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';
import { AuthContext } from './Authcontext';

const AuthPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { login } = useContext(AuthContext);

  useEffect(() => {
    if (location.state?.mode === 'signup') {
      setIsLogin(false);
    }
  }, [location.state]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = isLogin
        ? await axios.post('http://localhost:3000/login', { email, password })
        : await axios.post('http://localhost:3000/signup', { name, email, password });

      if (isLogin) {
        login(response.data.user, response.data.token);
        alert("Login successful");
        navigate('/')

      } else {
        alert("Signup successful! Please login.");
        setIsLogin(true);
      }
    } catch (err) {
      alert(err.response?.data?.message || "Something went wrong");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-500 to-indigo-600">
      <div className="w-[90%] max-w-4xl h-[500px] bg-white shadow-2xl rounded-xl flex overflow-hidden relative transition-all duration-700">
        <div className="w-3/5 flex items-center justify-center p-10">
          <form className="w-full max-w-sm space-y-4" onSubmit={handleSubmit}>
            <h2 className="text-3xl font-bold text-indigo-600 mb-6">
              {isLogin ? 'Login to your account' : 'Create your account'}
            </h2>
            {!isLogin && (
              <input
                type="text"
                placeholder="Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-md"
              />
            )}
            <input
              type="email"
              placeholder="Email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md"
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-md"
            />
            <button
              type="submit"
              className="w-full bg-indigo-600 text-white py-2 rounded-md hover:bg-indigo-700 transition"
            >
              {isLogin ? 'Login' : 'Sign Up'}
            </button>
          </form>
        </div>

        <div className="w-2/5 bg-indigo-600 text-white flex flex-col justify-center items-center p-10">
          <h2 className="text-2xl font-bold mb-4">
            {isLogin ? 'New here?' : 'Already have an account?'}
          </h2>
          <p className="mb-6 text-center text-sm md:text-base">
            {isLogin
              ? 'Sign up to create an account and get started!'
              : 'Login with your existing account credentials.'}
          </p>
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="bg-white text-indigo-600 px-6 py-2 rounded-full font-semibold hover:bg-gray-100 transition"
          >
            {isLogin ? 'Sign Up' : 'Login'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuthPage;
