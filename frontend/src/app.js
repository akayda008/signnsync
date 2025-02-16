const video = document.getElementById('video');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
let mediaRecorder;
let recordedChunks = [];

// Start video stream
navigator.mediaDevices.getUserMedia({ video: true, audio: true })
    .then(stream => {
        video.srcObject = stream;
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = event => recordedChunks.push(event.data);
    })
    .catch(err => console.error("Camera/Mic Access Error:", err));

// Start Recording
startBtn.addEventListener('click', () => {
    recordedChunks = [];
    mediaRecorder.start();
    console.log("Recording started...");
});

// Stop Recording & Send to Backend
stopBtn.addEventListener('click', () => {
    mediaRecorder.stop();
    console.log("Recording stopped...");

    mediaRecorder.onstop = async () => {
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const formData = new FormData();
        formData.append('video', blob);

        const response = await fetch('http://localhost:5000/api/video/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        console.log(result);
    };
});
