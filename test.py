from flask import Flask, jsonify
import threading
import cv2

app = Flask(__name__)

def eye_tracking():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try DirectShow backend
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame")
            break  # Exit loop if frame is not captured

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Stop on 'q' key press

    cap.release()
    cv2.destroyAllWindows()

@app.route('/run-eye-tracking', methods=['POST'])
def run_eye_tracking():
    thread = threading.Thread(target=eye_tracking)
    thread.daemon = True  # Allows Flask to exit even if thread is running
    thread.start()
    return jsonify({"message": "Eye Tracking started in background!"})

if __name__ == '__main__':
    app.run(debug=True)
