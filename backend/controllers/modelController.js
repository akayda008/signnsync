const fs = require('fs');
const path = require('path');

// Function to save video file
const saveVideo = (req, res) => {
    if (!req.file) {
        return res.status(400).json({ message: "No video uploaded!" });
    }

    const videoPath = path.join(__dirname, '../uploads/', req.file.filename);
    res.status(200).json({ message: "Video uploaded successfully!", filePath: videoPath });
};

module.exports = { saveVideo };
