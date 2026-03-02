import numpy as np
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------------------
# Load trained model
# ---------------------------
model = load_model("har_model.keras")

activity_labels = {
    0: "Walking",
    1: "Walking Upstairs",
    2: "Walking Downstairs",
    3: "Sitting",
    4: "Standing",
    5: "Laying"
}

# ---------------------------
# WEB PAGE
# ---------------------------
@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>HAR Live Detection</title>

<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#1e3c72">

<style>
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    text-align: center;
}

.container {
    background: rgba(255,255,255,0.1);
    padding: 40px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    width: 90%;
    max-width: 400px;
}

h1 { margin-bottom: 20px; }

.status {
    font-size: 22px;
    margin: 20px 0;
    font-weight: bold;
}

.confidence {
    font-size: 16px;
    opacity: 0.8;
}

button {
    background: #00c6ff;
    border: none;
    padding: 12px 25px;
    font-size: 18px;
    border-radius: 30px;
    cursor: pointer;
    transition: 0.3s;
    color: white;
}

button:hover {
    background: #0072ff;
    transform: scale(1.05);
}

.pulse {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.08); }
    100% { transform: scale(1); }
}
</style>

</head>
<body>

<div class="container">
    <h1>Human Activity Recognition</h1>
    <div id="status" class="status">Press Start</div>
    <div id="confidence" class="confidence"></div>
    <button onclick="startCollection()">Start Detection</button>
</div>

<script>

let sensorData = [];
let collecting = false;

async function requestPermission() {
    if (typeof DeviceMotionEvent.requestPermission === 'function') {
        try {
            await DeviceMotionEvent.requestPermission();
        } catch (e) {
            alert("Sensor permission denied");
        }
    }
}

async function startCollection() {

    await requestPermission();

    if (collecting) return;

    collecting = true;
    sensorData = [];
    document.getElementById("status").innerHTML = "Collecting sensor data...";
    document.getElementById("status").classList.add("pulse");

    window.addEventListener("devicemotion", function(event) {

        let ax = event.accelerationIncludingGravity.x || 0;
        let ay = event.accelerationIncludingGravity.y || 0;
        let az = event.accelerationIncludingGravity.z || 0;

        let gx = event.rotationRate?.alpha || 0;
        let gy = event.rotationRate?.beta || 0;
        let gz = event.rotationRate?.gamma || 0;

        sensorData.push([ax, ay, az, gx, gy, gz]);

        if (sensorData.length === 128) {
            sendData(sensorData);
            sensorData = [];
        }

    });
}

function sendData(data) {

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({data: data})
    })
    .then(response => response.json())
    .then(result => {

        document.getElementById("status").classList.remove("pulse");

        if (result.error) {
            document.getElementById("status").innerHTML = "Error";
            return;
        }

        document.getElementById("status").innerHTML =
            "Activity: " + result.predicted_activity;

        document.getElementById("confidence").innerHTML =
            "Confidence: " + result.confidence + "%";

    });
}

// Register service worker
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js');
}

</script>

</body>
</html>
""")

# ---------------------------
# Prediction API
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["data"]
        sensor_data = np.array(data)

        if sensor_data.shape != (128, 6):
            return jsonify({"error": "Invalid input shape"})

        sensor_data = np.expand_dims(sensor_data, axis=0)

        prediction = model.predict(sensor_data, verbose=0)

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "predicted_activity": activity_labels[class_index],
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------------
# Manifest (PWA)
# ---------------------------
@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "HAR Detection",
        "short_name": "HAR",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#1e3c72",
        "theme_color": "#1e3c72",
        "icons": [
            {
                "src": "https://cdn-icons-png.flaticon.com/512/2910/2910791.png",
                "sizes": "192x192",
                "type": "image/png"
            }
        ]
    })

# ---------------------------
# Service Worker
# ---------------------------
@app.route("/sw.js")
def sw():
    return """
self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open('har-cache').then(function(cache) {
      return cache.addAll(['/']);
    })
  );
});
"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)