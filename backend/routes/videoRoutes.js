const express = require('express');
const router = express.Router();

// Sample Route for Video Processing
router.post('/upload', (req, res) => {
    res.send({ message: "Video Received" });
});

module.exports = router;
