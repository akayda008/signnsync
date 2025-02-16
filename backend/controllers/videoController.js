async function analyzeVideo(videoData) {
    const response = await fetch('http://localhost:5000/api/model/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video: videoData })
    });
    const data = await response.json();
    console.log(data.result);
}
