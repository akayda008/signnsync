const express = require('express');
const router = express.Router();

// Connect to Flask API for ML Model Processing
router.post('/analyze', async (req, res) => {
    // Send data to Python Flask API
    res.send({ result: "Sign Language Recognized: Hello" });
});

module.exports = router;
