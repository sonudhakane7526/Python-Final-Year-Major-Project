// Start Webcam Feed
const video = document.getElementById("video");

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing webcam:", err);
    });

// Function to Display Feature Selection
function showFeature(feature) {
    document.getElementById("detection-message").innerText = `${feature.replace("-", " ").toUpperCase()} is now active.`;
}

