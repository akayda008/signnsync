const express = require('express');
const cors = require('cors');

const app = express(); // Initialize Express

app.use(cors()); // Enable CORS
app.use(express.json()); // Allow JSON request bodies

// Import Routes
const videoRoutes = require('./routes/videoRoutes');
const modelRoutes = require('./routes/modelRoutes');

app.use('/api/video', videoRoutes);
app.use('/api/model', modelRoutes);

// Define PORT and start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
