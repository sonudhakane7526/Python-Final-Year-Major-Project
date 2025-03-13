# import cv2
# import numpy as np
# import threading

# # Import your modules (update with actual implementations)
# from eye_tracker import track_eyes
# from head_pose_estimation import estimate_head_pose
# from mouth_opening_detector import detect_mouth_opening
# from audio_part import monitor_audio
# from face_detector import detect_faces
# from person_phone import detect_person_phone
# from face_landmarks import detect_face_landmarks
# from face_spoofing import detect_face_spoofing

# # Load models once to optimize performance
# face_model = detect_faces()  # Update with actual model loading code

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# def process_frame(frame):
#     """Run all feature detection modules on a single frame."""
#     results = {}

#     # Run each feature in a separate thread for better performance
#     threads = []

#     def run_eye_tracker():
#         results['eye_tracking'] = track_eyes(frame)

#     def run_head_pose():
#         results['head_pose'] = estimate_head_pose(frame)

#     def run_mouth_opening():
#         results['mouth_opening'] = detect_mouth_opening(frame)

#     def run_audio_monitor():
#         results['audio'] = monitor_audio()

#     def run_face_detector():
#         results['face_detection'] = detect_faces(frame, face_model)

#     def run_person_phone():
#         results['person_phone'] = detect_person_phone(frame)

#     def run_face_landmarks():
#         results['face_landmarks'] = detect_face_landmarks(frame)

#     def run_face_spoofing():
#         results['face_spoofing'] = detect_face_spoofing(frame)

#     # Start threads
#     for func in [run_eye_tracker, run_head_pose, run_mouth_opening, 
#                  run_audio_monitor, run_face_detector, run_person_phone,
#                  run_face_landmarks, run_face_spoofing]:
#         thread = threading.Thread(target=func)
#         threads.append(thread)
#         thread.start()

#     # Wait for all threads to complete
#     for thread in threads:
#         thread.join()

#     return results

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Process frame through all modules
#     detections = process_frame(frame)

#     # Display results (customize as needed)
#     for key, value in detections.items():
#         print(f"{key}: {value}")

#     # Show the frame
#     cv2.imshow("Online Exam Monitoring", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()












# import cv2
# import mediapipe as mp
# import numpy as np
# import dlib
# import time
# import pyaudio
# import wave
# import threading

# # Initialize MediaPipe Solutions
# mp_face_mesh = mp.solutions.face_mesh
# mp_face_detection = mp.solutions.face_detection
# mp_face = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Initialize Dlib Face Detector
# face_detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Initialize OpenCV Pre-trained Model for Person & Phone Detection
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Audio Monitoring Function
# def record_audio():
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 1
#     RATE = 44100
#     RECORD_SECONDS = 10
#     WAVE_OUTPUT_FILENAME = "audio_output.wav"

#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT, channels=CHANNELS,
#                         rate=RATE, input=True,
#                         frames_per_buffer=CHUNK)
#     frames = []

#     print("Recording Audio...")
#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)

#     print("Finished Recording")
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()

#     waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#     waveFile.setnchannels(CHANNELS)
#     waveFile.setsampwidth(audio.get_sample_size(FORMAT))
#     waveFile.setframerate(RATE)
#     waveFile.writeframes(b''.join(frames))
#     waveFile.close()

# # Start Audio Thread
# audio_thread = threading.Thread(target=record_audio)
# audio_thread.start()

# # Start Video Capture
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     h, w, _ = frame.shape
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Face Detection
#     faces = face_detector(gray)
#     for face in faces:
#         x, y, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
#         cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

#     # Face Mesh Processing
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = mp_face.process(rgb_frame)
#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             for landmark in face_landmarks.landmark:
#                 x = int(landmark.x * w)
#                 y = int(landmark.y * h)
#                 cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

#     # YOLO Object Detection (Person & Phone)
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outputs = net.forward(output_layers)
#     for output in outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 center_x = int(detection[0] * w)
#                 center_y = int(detection[1] * h)
#                 width = int(detection[2] * w)
#                 height = int(detection[3] * h)
#                 x = int(center_x - width / 2)
#                 y = int(center_y - height / 2)
#                 color = (0, 255, 0) if classes[class_id] == "person" else (0, 0, 255)
#                 cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
#                 label = f"{classes[class_id]}: {confidence:.2f}"
#                 cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Display the Frame
#     cv2.imshow("Integrated Proctoring AI", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import asyncio
import websockets
import json

# Load pre-trained YOLO model (for object detection)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

async def video_stream(websocket, path):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        # YOLO Object Detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    label = classes[class_id]

                    detections.append({"label": label, "x": x, "y": y, "w": w, "h": h})

        # Send detections as JSON
        await websocket.send(json.dumps(detections))

    cap.release()

# Start WebSocket Server
start_server = websockets.serve(video_stream, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
