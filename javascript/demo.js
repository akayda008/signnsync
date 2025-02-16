let video = document.getElementById("video");

// Access webcam and mic
navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing webcam/mic: ", error);
    });

// Function to capture frame and send to backend
function captureAndSend(endpoint) {
    let canvas = document.getElementById("canvas");
    let context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    let dataURL = canvas.toDataURL("image/jpeg");

    fetch(endpoint, {
        method: "POST",
        body: JSON.stringify({ image: dataURL }),
        headers: { "Content-Type": "application/json" }
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = data.output;
    })
    .catch(error => console.error("Error:", error));
}

// Mode Selection
function startSignLanguage() {
    captureAndSend("/sign_language");
}

function startEmotionRecognition() {
    captureAndSend("/emotion_detection");
}

function startBoth() {
    captureAndSend("/both");
}
