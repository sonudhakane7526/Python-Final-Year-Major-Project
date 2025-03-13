from flask import Flask, Response, request, jsonify, render_template
import cv2
import numpy as np

app = Flask(__name__)

# Load OpenCV's pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start Video Capture
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Face Detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('webpage.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_feature', methods=['POST'])
def detect_feature():
    data = request.json
    feature = data.get("feature")
    
    if feature == "face-detection":
        return jsonify({"status": "Face Detection activated"})
    elif feature == "eye-tracking":
        return jsonify({"status": "Eye Tracking activated (Coming soon)"})
    elif feature == "head-pose":
        return jsonify({"status": "Head Pose Detection activated (Coming soon)"})
    elif feature == "mouth-opening":
        return jsonify({"status": "Mouth Opening Detection activated (Coming soon)"})
    elif feature == "person-phone":
        return jsonify({"status": "Person and Phone Detection activated (Coming soon)"})
    elif feature == "audio-monitoring":
        return jsonify({"status": "Audio Monitoring activated (Coming soon)"})
    
    return jsonify({"error": "Invalid feature"}), 400

if __name__ == "__main__":
    app.run(debug=True)