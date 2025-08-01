import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../App.css'; 
import apiService from '../config/api';
import { errorHandler } from '../utils/errorHandler';

const ChangeMapView = ({ coords }) => {
  const map = useMap();
  map.flyTo(coords, 14);
  return null;
};

const Home = () => {
  const [activePopup, setActivePopup] = useState(null);
  const [infoPopup, setInfoPopup] = useState(false);
  const [futurePrediction, setFuturePrediction] = useState(false);
  const [address, setAddress] = useState('');
  const [position, setPosition] = useState([13.0827, 80.2707]);
  const [seaData, setSeaData] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [safetyMessage, setSafetyMessage] = useState(null);
  const [alertMessage, setAlertMessage] = useState(null);
  const localUser = JSON.parse(localStorage.getItem("user"));
  const [popupContent, setPopupContent] = useState({
    sea: {
      title: 'Current Sea Conditions',
      message: 'Loading...',
    },
    weather: {
      title: 'Weather Forecast',
      message: 'Loading...',
    },
    alert: {
      title: 'Safety Alert',
      message: 'Loading...',
    },
    lifeguard: {
      title: 'Nearby Lifeguards',
      message: 'Loading...',
    },
  });

  const [lat, setLat] = useState(null);
  const [lon, setLon] = useState(null);
  const [FutureData, setFutureData] = useState(null);
  useEffect(() => {
    const fetchSeaAndWeatherData = async () => {
      try {
        const [lat, lon] = position;
        setLat(lat);
        setLon(lon);

        const [marineRes, weatherRes] = await Promise.all([
          apiService.getMarineData(lat, lon),
          apiService.getWeatherData(lat, lon),
        ]);

        setSeaData(marineRes.current);
        setWeatherData(weatherRes.current);
      } catch (error) {
        errorHandler.logError(error, { context: 'fetch_weather_data', lat, lon });
        errorHandler.showError(error);
        setSeaData(null);
        setWeatherData(null);
      }
    };

    fetchSeaAndWeatherData();
  }, [position]);

  // Handle location search and prediction
  const handleSearch = async () => {
    
    try {
      const response = await apiService.geocodeLocation(address);
      if (response.length > 0) {
        const { lat, lon } = response[0];
        const newPos = [parseFloat(lat), parseFloat(lon)];
        setPosition(newPos);
        setTimeout(async () => {
          if (seaData && weatherData) {
            console.log(localUser);
            const user_mail = localUser.email; 

            const payload = {
              user_mail,
              BeachName: address,
              latitude: newPos[0],
              longitude: newPos[1],
              Temperature:weatherData.temperature_2m,
              Humidity:weatherData.relative_humidity_2m,
              WindSpeed:weatherData.wind_speed_10m,
              CloudCover:weatherData.cloud_cover,
              WeatherCode:weatherData.weather_code,
              WaveHeight:seaData.wave_height,
              OceanCurrentVelocity:seaData.ocean_current_velocity,
              SeaSurfaceTemp:seaData.sea_surface_temperature,
            };

            try {
              const predictRes = await apiService.makePrediction(payload);
              const { alert_message, safety_message } = predictRes;
              setSafetyMessage(safety_message);
              setAlertMessage(alert_message);
              errorHandler.showSuccess('Safety prediction updated successfully');
            } catch (error) {
              errorHandler.logError(error, { context: 'prediction_api', payload });
              errorHandler.showError(error);
              setSafetyMessage('Unable to get safety prediction. Please try again.');
              setAlertMessage('Service temporarily unavailable.');
            }
              setPopupContent({
                sea: {
                  title: 'Current Sea Conditions',
                  message: `Wave Height: ${seaData.wave_height || 'N/A'}m, Tide: ${seaData.ocean_current_velocity > 0.5 ? 'Rising' : 'Calm'}, Water Temp: ${seaData.sea_surface_temperature || 'N/A'}°C`,
                },
                weather: {
                  title: 'Weather Forecast',
                  message: `Temp: ${weatherData.temperature_2m || 'N/A'}°C, Humidity: ${weatherData.relative_humidity_2m || 'N/A'}%, Wind: ${weatherData.wind_speed_10m || 'N/A'} km/h, Cloud Cover: ${weatherData.cloud_cover || 'N/A'}%`,
                },
                alert: {
                  title: 'Safety Alert',
                  message: alert_message || 'Stay updated on conditions.',
                },
                lifeguard: {
                  title: 'Nearby Lifeguards',
                  message: 'Lifeguard station is 500m south of your location.',
                }
              });
            } catch (error) {
              errorHandler.logError(error, { context: 'prediction_api', payload });
              errorHandler.showError(error);
            }
          }
        }, 2000); 
      } else {
        errorHandler.showError(new Error('Location not found. Please try a different search term.'));
      }
    } catch (error) {
      errorHandler.logError(error, { context: 'geocoding', address });
      errorHandler.showError(error);
    }
  };
  
  const openPopup = (type) => setActivePopup(type);
  const closePopup = () => setActivePopup(null);
  const openInfo = () => setInfoPopup(true);
  const closeInfo = () => setInfoPopup(false);
  const closeFuture = () => setFuturePrediction(false);
  const openFuturePrediction = async () => {
    try {
      const data = await apiService.getForecast(lat, lon);
      console.log("Forecast:", data);
      setFutureData(data.forecast);
      errorHandler.showSuccess('Future forecast loaded successfully');
  
      setFuturePrediction(true);
    } catch (error) {
      errorHandler.logError(error, { context: 'future_prediction', lat, lon });
      errorHandler.showError(error);
    }
  };
  

  return (
    <div className="p-4 font-sans bg-gradient-to-b from-blue-100 via-white to-blue-50 min-h-screen relative">
      <div>
      {/* Search Bar */}
      <div className="w-full flex justify-center mb-4">
        <input
          type="text"
          placeholder="Search for a beach or location..."
          value={address}
          onChange={(e) => setAddress(e.target.value)}
          className="w-full max-w-xl px-4 py-2 border rounded shadow-sm focus:outline-none"
        />
        <button
          onClick={handleSearch}
          className="ml-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Search
        </button>
      </div>

      {/* Map */}
      <div className="relative z-10 w-full h-[400px] mb-6 rounded overflow-hidden shadow-lg">
        <MapContainer center={position} zoom={10} className="h-full w-full leaflet-container">
          <TileLayer
            attribution="© OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
           <Marker position={position}>
            <Popup>You are here!</Popup>
          </Marker>
          <ChangeMapView coords={position} />
        </MapContainer>
      </div>
    </div>
    {/* Info and Future Prediction Section */}
    <div className="flex justify-between items-center mb-4 px-2">
      <div className="text-xl text-gray-600">Quick Access</div>
      
      {/* Buttons Container */}
      <div className="ml-auto flex gap-4">
        {/* Info Button */}
        <button
          onClick={openInfo}
          className="bg-blue-500 text-white px-4 py-2 rounded-full hover:bg-blue-600"
        >
          Info
        </button>
        
        {/* Future Prediction Button */}
        <button
          onClick={openFuturePrediction}
          className="bg-green-500 text-white px-4 py-2 rounded-full hover:bg-green-600"
        >
          Future Prediction
        </button>
      </div>
    </div>
      {/* Feature Buttons */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <button onClick={() => openPopup('sea')} className="bg-cyan-600 text-white py-3 rounded-md shadow hover:bg-cyan-700">Sea Conditions</button>
        <button onClick={() => openPopup('weather')} className="bg-indigo-600 text-white py-3 rounded-md shadow hover:bg-indigo-700">Weather Forecast</button>
        <button onClick={() => openPopup('alert')} className="bg-orange-600 text-white py-3 rounded-md shadow hover:bg-orange-700">Safety Alerts</button>
        <button onClick={() => openPopup('lifeguard')} className="bg-teal-600 text-white py-3 rounded-md shadow hover:bg-teal-700">Nearby Lifeguards</button>
      </div>

      {/* Feature Popup Modal */}
      {activePopup === 'lifeguard' ? (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-60 flex justify-center items-center">
          <div className="bg-white rounded-2xl p-4 max-w-md w-11/12 shadow-xl relative">
            <button onClick={closePopup} className="absolute top-2 right-3 text-gray-600 text-2xl">&times;</button>
            <h2 className="text-lg font-bold mb-2 text-center">Drishti Lifesaving Pvt Ltd</h2>
            <img
              src="https://lh5.googleusercontent.com/p/AF1QipNMGwdl0AoGdEVjDtdRnlgabV9k3r8q5uEUFgL3=w408-h306-k-no"
              alt="Lifeguard Station"
              className="rounded-lg mb-3 w-full h-40 object-cover"
            />
            <div className="text-sm text-gray-800 flex items-start gap-2 mb-2">
              <p>6WWF+43M, Unnamed Road, Chennai 403716</p>
            </div>
            <div className="text-sm text-gray-800 flex items-center gap-2 mb-2">
              <a href="tel:08008331511" className="underline">08008331511</a>
            </div>
            <div className="text-sm text-gray-800 flex items-center gap-2">
              <a href="#" className="underline">Message</a>
            </div>
          </div>
        </div>
      ) : activePopup && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex justify-center items-center">
          <div className="bg-white p-6 rounded-lg w-11/12 max-w-md text-center shadow-lg relative z-50">
            <h2 className="text-xl font-bold mb-2">{popupContent[activePopup].title}</h2>
            <p className="text-gray-700 mb-4">{popupContent[activePopup].message}</p>
            <button
              onClick={closePopup}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Info Popup Modal */}
      {infoPopup && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-60 flex justify-center items-center">
          <div className="bg-white p-6 rounded-lg w-11/12 max-w-2xl text-left shadow-lg relative z-50 overflow-y-auto max-h-[80vh]">
            <h2 className="text-2xl font-bold mb-4 text-blue-700">Beach Safety Information</h2>

            <h3 className="font-semibold mb-1">Beach Name:</h3>
            <p className="mb-4 capitalize">{address}</p>

            <h3 className="font-semibold mb-1">Sea Conditions:</h3>
            <p className="mb-4">
              Wave Height: {seaData.wave_height} m | Ocean Current: {seaData.ocean_current_velocity} m/s | 
              Sea Surface Temp: {seaData.sea_surface_temperature} °C
            </p>

            <h3 className="font-semibold mb-1">Weather Forecast:</h3>
            <p className="mb-4">
              Temperature: {weatherData.temperature_2m} °C | Humidity: {weatherData.relative_humidity_2m}% | Wind Speed: {weatherData.wind_speed_10m} km/h | Cloud Cover: {weatherData.cloud_cover}%
            </p>

            <h3 className="font-semibold mb-1">Alerts:</h3>
            <p className="mb-4">{alertMessage}</p>

            <h3 className="font-semibold mb-1 text-red-600">Safety Measures:</h3>
            <ul className="list-disc list-inside mb-4 text-gray-700">
              <li>{safetyMessage}</li>
            </ul>

            <div className="text-center">
              <button
                onClick={closeInfo}
                className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Future data Popup Modal */}
      {futurePrediction && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-60 flex justify-center items-center">
          <div className="bg-white p-6 rounded-lg w-11/12 max-w-3xl text-left shadow-lg relative z-50 overflow-y-auto max-h-[80vh]">
            <h2 className="text-2xl font-bold mb-4 text-blue-700 text-center">Beach Safety Forecast (Next 3 Days)</h2>
            
            <div className="space-y-6">
              {FutureData?.map((data, index) => (
                <div key={index} className="border p-4 rounded shadow-md bg-blue-50">
                  <h3 className="font-semibold text-lg mb-2">Day {data.day}</h3>
                  <p><strong>Avg Temperature:</strong> {data.avg_temp}°C</p>
                  <p><strong>Avg Wind Direction:</strong> {data.avg_wind_direction}</p>
                  <p><strong>Avg Wind Speed:</strong> {data.avg_wind_speed} km/h</p>
                  <p><strong>Wave Height:</strong> {data.avg_wave_height} m</p>
                  <p><strong>Sea Surface Temp:</strong> {data.avg_sea_surface_temp}°C</p>
                  <p><strong>Beach Safety:</strong> {data.beach_safety}</p>
                </div>
              ))}
              </div>

              <div className="text-center mt-6">
                <button
                  onClick={closeFuture}
                  className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                >
                  Close
                </button>
              </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
