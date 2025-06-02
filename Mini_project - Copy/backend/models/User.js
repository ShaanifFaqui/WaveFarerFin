const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: false
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  password: {
    type: String,
    required: true
  },
  mobile: {
    type: String,
    required: false
  },
  location: {
    type: String,
    required: false
  },
  preferedBeach: {
    type: String,
    required: false
  },
  Language: {
    type: String,
    required: false
  }

});

module.exports = mongoose.model('User', userSchema);
