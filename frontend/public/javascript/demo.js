const video = document.getElementById("video");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const resultDiv = document.getElementById("result");

let mediaRecorder;
let recordedChunks = [];
let taskType = "emotion"; // Default task type

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

        mediaRecorder.onstop = async () => {
            const blob = new Blob(recordedChunks, { type: "video/webm" });
            recordedChunks = [];

            // Convert WebM to MP4 before sending
            const mp4Blob = await convertWebMToMP4(blob);
            sendVideoToServer(mp4Blob);
        };
    })
    .catch((error) => {
        console.error("❌ Error accessing webcam:", error);
    });

// Start recording
startButton.addEventListener("click", () => {
    recordedChunks = [];
    mediaRecorder.start();
    console.log(`🎥 Recording started for: ${taskType}`);
});

// Stop recording
stopButton.addEventListener("click", () => {
    mediaRecorder.stop();
    console.log(`🛑 Recording stopped. Sending to server for: ${taskType}`);
});

// Function to set the task type
function startRecognition(selectedTask) {
    taskType = selectedTask;
    console.log(`🔹 Selected Task: ${taskType}`);
}

// Convert WebM to MP4
async function convertWebMToMP4(webmBlob) {
    const file = new File([webmBlob], "video.mp4", { type: "video/mp4" });
    return file;
}

// Send video to Flask backend
function sendVideoToServer(videoBlob) {
    const formData = new FormData();
    formData.append("video", videoBlob, "video.mp4");

    fetch(`http://127.0.0.1:5000/predict/${taskType}`, {
        method: "POST",
        body: formData,
    })
    .then((response) => response.json())
    .then((data) => {
        console.log("✅ Server Response:", data);
        displayResult(data);
    })
    .catch((error) => {
        console.error("❌ Error:", error);
        resultDiv.innerHTML = `<p style="color: red;">❌ Error processing video. Please try again.</p>`;
    });
}

// Display predictions
function displayResult(data) {
    let resultHTML = `<h3>🔹 Prediction Results:</h3>`;

    if (taskType === "emotion") {
        resultHTML += `<p><strong>Emotion:</strong> ${data.emotion ?? "No face detected"}</p>`;
    } else if (taskType === "sign") {
        resultHTML += `<p><strong>Left Hand:</strong> ${data.left_hand ?? "No left hand detected"}</p>`;
        resultHTML += `<p><strong>Right Hand:</strong> ${data.right_hand ?? "No right hand detected"}</p>`;
    } else if (taskType === "both") {
        resultHTML += `<p><strong>Emotion:</strong> ${data.emotion ?? "No face detected"}</p>`;
        resultHTML += `<p><strong>Left Hand:</strong> ${data.left_hand ?? "No left hand detected"}</p>`;
        resultHTML += `<p><strong>Right Hand:</strong> ${data.right_hand ?? "No right hand detected"}</p>`;
    }

    resultDiv.innerHTML = resultHTML;
}
