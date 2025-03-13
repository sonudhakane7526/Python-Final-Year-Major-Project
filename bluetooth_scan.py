# import cv2
# import torch

# # Load the pre-trained YOLOv5 model
# model = torch.hub.load('ultralytics/yolov3', 'yolov3s')  # Use 'yolov3s' for small, 'yolov3m' for medium, etc.

# # Open a connection to the webcam (or use a video file)
# cap = cv2.VideoCapture(0)  # 0 for webcam, or provide video file path

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
   
#     # Perform object detection
#     results = model(frame)
   
#     # Draw bounding boxes and labels on the frame
#     results.render()
   
#     # Display the frame with detections
#     cv2.imshow('Phone Detection', frame)
   
#     # Exit loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load YOLO
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Load class names
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]

# def detect_objects(frame):
#     height, width, channels = frame.shape

#     # Prepare the frame for YOLO
#     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     outs = net.forward(output_layers)

#     # Process detections
#     class_ids = []
#     confidences = []
#     boxes = []

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:  # Threshold for detection
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     # Apply Non-Maximum Suppression
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     if len(indexes) > 0:
#         for i in indexes.flatten():  # Flatten the array if necessary
#             box = boxes[i]
#             x, y, w, h = box
#             label = str(classes[class_ids[i]])
#             confidence = confidences[i]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     return frame

# # Open video capture
# cap = cv2.VideoCapture(0)  # 0 for webcam, or provide video file path

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect objects
#     frame = detect_objects(frame)

#     # Convert frame to RGB (for displaying with matplotlib)
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Display the frame with detections using matplotlib
#     plt.imshow(frame_rgb)
#     plt.axis('off')  # Turn off axis numbers and ticks
#     plt.show()

#     # Optional: add a break condition if you want to stop after displaying one frame
#     break

# cap.release()


import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_objects(frame):
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        print("Output shape:", out.shape)  # Debugging: Print shape of output
        for detection in out:
            for obj in detection:
                obj = obj.reshape(-1)  # Flatten the detection
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Threshold for detection
                    center_x = int(obj[0] * width)
                    center_y = int(obj[1] * height)
                    w = int(obj[2] * width)
                    h = int(obj[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    # Apply Non-Maximum Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():  # Flatten the array if necessary
            box = boxes[i]
            x, y, w, h = box
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Open video capture
cap = cv2.VideoCapture(0)  # 0 for webcam, or provide video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    frame = detect_objects(frame)

    # Display the frame with detections
    cv2.imshow('YOLOv3 Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
