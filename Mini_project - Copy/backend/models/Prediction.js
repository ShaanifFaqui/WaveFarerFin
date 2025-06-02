const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  timestamp: {
    type: Date,
    default: Date.now,
  },
  email: {
    type: String,
    required: true,
  },
  BeachName: {
    type: String,
    required: true,
  },
  input: {
    user_mail: { type: String, required: true },
    BeachName: { type: String, required: true },
    latitude: { type: Number, required: true },
    longitude: { type: Number, required: true },
    Temperature: { type: Number, required: true },
    Humidity: { type: Number, required: true },
    WindSpeed: { type: Number, required: true },
    CloudCover: { type: Number, required: true },
    WeatherCode: { type: Number, required: true }, 
    WaveHeight: { type: Number, required: true },
    OceanCurrentVelocity: { type: Number, required: true },
    SeaSurfaceTemp: { type: Number, required: true },
  },
  prediction: {
    alert_message: { type: String, required: true },
    safety_message: { type: String, required: true },
  },
});

module.exports = mongoose.model('Prediction', PredictionSchema);
