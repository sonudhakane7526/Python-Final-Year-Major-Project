import cv2
import numpy as np

def get_face_detector(modelFile=None,
                      configFile=None,
                      quantized=False):
    if quantized:
        if modelFile == None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile == None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    else:
        if modelFile == None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile == None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def draw_faces(img, faces):
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)


import cv2
import numpy as np

def get_face_detector(modelFile=None, configFile=None, quantized=False):
    if quantized:
        if modelFile is None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile is None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    else:
        if modelFile is None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile is None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model


def find_faces(img, model):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces


def draw_faces(img, faces):
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)


def main():
    # Initialize the face detector
    face_model = get_face_detector()

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        faces = find_faces(frame, face_model)
        draw_faces(frame, faces)

        # Display the output
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import cv2
import numpy as np

def get_face_detector(modelFile=None, configFile=None, quantized=False):
    if quantized:
        if modelFile is None:
            modelFile = "models/opencv_face_detector_uint8.pb"
        if configFile is None:
            configFile = "models/opencv_face_detector.pbtxt"
        model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    else:
        if modelFile is None:
            modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
        if configFile is None:
            configFile = "models/deploy.prototxt"
        model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return model

def find_faces(img, model):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    res = model.forward()
    faces = []
    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence > 0.5:
            box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append([x, y, x1, y1])
    return faces

def draw_faces(img, faces):
    for x, y, x1, y1 in faces:
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 3)

def main():
    # Initialize the face detector
    face_model = get_face_detector()

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        faces = find_faces(frame, face_model)
        draw_faces(frame, faces)

        # Display the output
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# import cv2
# import numpy as np

# def get_face_detector(modelFile=None, configFile=None, quantized=False):
#     if quantized:
#         if modelFile is None:
#             modelFile = "models/opencv_face_detector_uint8.pb"
#         if configFile is None:
#             configFile = "models/opencv_face_detector.pbtxt"
#         model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
#     else:
#         if modelFile is None:
#             modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
#         if configFile is None:
#             configFile = "models/deploy.prototxt"
#         model = cv2.dnn.readNetFromCaffe(configFile, modelFile)
#     return model

# def find_faces(img, model):
#     h, w = img.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#     model.setInput(blob)
#     res = model.forward()
#     faces = []
#     for i in range(res.shape[2]):
#         confidence = res[0, 0, i, 2]
#         if confidence > 0.5:
#             box = res[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x, y, x1, y1) = box.astype("int")
#             faces.append([x, y, x1, y1])
#     return faces

# def draw_faces(img, faces):
#     for x, y, x1, y1 in faces:
#         # Calculate the center and radius for the circle
#         center_x = (x + x1) // 2
#         center_y = (y + y1) // 2
#         radius = int(((x1 - x) + (y1 - y)) // 4)  # Adjust radius as needed
        
#         # Draw the circle
#         cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 3)

# def main():
#     # Initialize the face detector
#     face_model = get_face_detector()

#     # Capture video from the webcam
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame.")
#             break

#         faces = find_faces(frame, face_model)
#         draw_faces(frame, faces)

#         # Display the output
#         cv2.imshow('Face Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
