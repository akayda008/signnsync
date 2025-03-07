const video = document.getElementById("video");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const selectTask = document.getElementById("select-task");
const resultDiv = document.getElementById("result");
let mediaRecorder;
let recordedChunks = [];

// Start video streaming
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then((stream) => {
        video.srcObject = stream;
        mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: "video/webm" });
            recordedChunks = [];
            sendVideoToServer(blob);
        };
    })
    .catch((error) => {
        console.error("❌ Error accessing webcam:", error);
    });

// Start recording
startButton.addEventListener("click", () => {
    recordedChunks = [];
    mediaRecorder.start();
    console.log("🎥 Recording started...");
});

// Stop recording
stopButton.addEventListener("click", () => {
    mediaRecorder.stop();
    console.log("🛑 Recording stopped. Sending to server...");
});

// Send video to Flask backend
function sendVideoToServer(videoBlob) {
    const formData = new FormData();
    formData.append("video", videoBlob, "video.webm");

    const selectedTask = selectTask.value;
    let endpoint = "/predict/emotion"; // Default

    if (selectedTask === "sign") {
        endpoint = "/predict/sign";
    } else if (selectedTask === "both") {
        endpoint = "/predict/both";
    }

    fetch(`http://127.0.0.1:5000${endpoint}`, {
        method: "POST",
        body: formData,
    })
    .then((response) => response.json())
    .then((data) => {
        displayResult(data);
    })
    .catch((error) => {
        console.error("❌ Error:", error);
    });
}

// Display predictions
function displayResult(data) {
    resultDiv.innerHTML = `<h3>🔹 Prediction Results:</h3><pre>${JSON.stringify(data, null, 2)}</pre>`;
}

// Functions for task selection
function startSignLanguage() {
    selectTask.value = "sign";
    console.log("🔹 Selected: Sign Language");
}

function startEmotionRecognition() {
    selectTask.value = "emotion";
    console.log("🔹 Selected: Emotion Detection");
}

function startBoth() {
    selectTask.value = "both";
    console.log("🔹 Selected: Both");
}
