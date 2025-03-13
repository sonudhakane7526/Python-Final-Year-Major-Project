function runFeature(feature) {
    fetch('http://127.0.0.1:5000/run-eye-tracking', {  // Flask API URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ feature: feature })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('detection-message').innerText = data.message || data.error;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('detection-message').innerText = 'An error occurred!';
    });
}
