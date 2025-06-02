const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const dotenv = require('dotenv');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const axios = require('axios');

dotenv.config();
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());


// MongoDB connection
mongoose.connect(process.env.MONGODB_URL)
.then(() => console.log("MongoDB connected"))
.catch(err => console.error(err));

//  model
const User = require('./models/User');
const Prediction = require('./models/Prediction'); 

// Signup Route
app.post('/signup', async (req, res) => {
  const { name, email, password } = req.body;
  try {
    const existingUser = await User.findOne({ email });
    if (existingUser) return res.status(400).json({ message: 'User already exists' });

    const hashedPassword = await bcrypt.hash(password, 10);
    const user = new User({ name, email, password: hashedPassword });
    await user.save();
    res.status(201).json({ message: 'Signup successful' });
  } catch (err) {
    res.status(500).json({ error: 'Server error' });
  }
});

// Login Route
app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    const user = await User.findOne({ email });
    if (!user) return res.status(400).json({ message: 'Invalid credentials' });

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ message: 'Invalid credentials' });

    // Create JWT token
    const token = jwt.sign({ id: user._id, email: user.email }, process.env.JWT_SECRET, {
      expiresIn: '1h',
    });

    res.status(200).json({
      message: 'Login successful',
      token,
      user: { name: user.name, email: user.email }
    });
  } catch (err) {
    res.status(500).json({ error: 'Server error' });
  }
});

app.put('/update-profile/:email', async (req, res) => {
  try {
    const updatedUser = await User.findOneAndUpdate(
      { email: req.params.email },
      {
        mobile: req.body.phone,
        location: req.body.location,
        preferedBeach: req.body.preferredBeaches,
        Language: req.body.language,
      },
      { new: true }
    );
    res.json(updatedUser);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to update profile' });
  }
});

app.get('/get-user/:email', async (req, res) => {
  const { email } = req.params;
  try {
    const user = await User.findOne({ email });
    if (!user) return res.status(404).json({ message: "User not found" });
    res.status(200).json(user);
  } catch (error) {
    res.status(500).json({ error: "Server error" });
  }
});

//to fetch the current prediction data
app.get('/api/fetchdata', async (req, res) => {
  const { userEmail, beachName } = req.query;
  const BeachName = decodeURIComponent(beachName).replace(/\+/g, ' ')
  if (!userEmail || !beachName) {
    return res.status(400).json({ error: 'userEmail and beachName are required' });
  }

  // Calculate ±2 minutes time range
  const now = new Date();
  const startTime = new Date(now.getTime() - 2 * 60 * 1000); // 2 minutes ago
  const endTime = new Date(now.getTime() + 2 * 60 * 1000);   // 2 minutes 

  try {
    const data = await Prediction.findOne({
      BeachName: BeachName,
      email: userEmail,
      timestamp: { $gte: startTime, $lte: endTime }
    }).sort({ timestamp: -1 }); // in case there are multiple, get the latest

    if (!data) {
      return res.status(404).json({ message: 'No recent prediction found' });
    }

    res.json(data);
  } catch (error) {
    console.error('Error fetching prediction:', error);
    res.status(500).send('Server error while fetching prediction');
  }
});

// Start server
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
