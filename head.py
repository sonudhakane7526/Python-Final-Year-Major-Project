# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import time

# # # Initialize MediaPipe FaceMesh
# # mp_face_mesh = mp.solutions.face_mesh
# # face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # mp_drawing = mp.solutions.drawing_utils
# # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# # cap = cv2.VideoCapture(0)

# # while cap.isOpened():
# #     success, image = cap.read()
# #     if not success:
# #         print("Ignoring empty camera frame.")
# #         continue

# #     start = time.time()

# #     # Flip the image horizontally for a later selfie-view display
# #     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
# #     image.flags.writeable = False

# #     # Process the image and get the results
# #     results = face_mesh.process(image)

# #     image.flags.writeable = True
# #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# #     img_h, img_w, img_c = image.shape
# #     face_3d = []
# #     face_2d = []

# #     if results.multi_face_landmarks:
# #         for face_landmarks in results.multi_face_landmarks:
# #             face_2d = []
# #             face_3d = []
# #             for idx, lm in enumerate(face_landmarks.landmark):
# #                 if idx in [33, 263, 1, 61, 291, 199]:
# #                     x, y = int(lm.x * img_w), int(lm.y * img_h)
# #                     z = lm.z * 3000

# #                     face_2d.append([x, y])
# #                     face_3d.append([x, y, z])

# #             # Convert to numpy arrays
# #             face_2d = np.array(face_2d, dtype=np.float64)
# #             face_3d = np.array(face_3d, dtype=np.float64)

# #             # Camera matrix
# #             focal_length = 1 * img_w
# #             cam_matrix = np.array([[focal_length, 0, img_w / 2],
# #                                    [0, focal_length, img_h / 2],
# #                                    [0, 0, 1]])

# #             dist_matrix = np.zeros((4, 1), dtype=np.float64)

# #             # SolvePnP
# #             success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

# #             # Get rotation matrix and angles
# #             if success:
# #                 rmat, _ = cv2.Rodrigues(rot_vec)
# #                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

# #                 X, y, Z = angles * 360

# #                 # Determine head pose
# #                 if y < -10:
# #                     text = "Looking Left"
# #                 elif y > 10:
# #                     text = "Looking Right"
# #                 elif X < -10:
# #                     text = "Looking Down"
# #                 elif X > 10:
# #                     text = "Looking Up"
# #                 else:
# #                     text = "Forward"

# #                 # Draw the landmarks
# #                 for (x, y) in face_2d:
# #                     cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)

# #                 # Draw pose direction and angles
# #                 cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
# #                 cv2.putText(image, f"X: {X:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #                 cv2.putText(image, f"Y: {y:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# #                 cv2.putText(image, f"Z: {Z:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# #             end = time.time()
# #             fps = 1 / (end - start)
# #             cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

# #             # Draw face mesh landmarks
# #             mp_drawing.draw_landmarks(
# #                 image=image,
# #                 landmark_list=face_landmarks,
# #                 connections=mp_face_mesh.FACE_CONNECTIONS,
# #                 landmark_drawing_spec=drawing_spec,
# #                 connection_drawing_spec=drawing_spec)

# #     cv2.imshow('Head Pose Estimation', image)

# #     if cv2.waitKey(5) & 0xFF == 27:
# #         break

# # cap.release()
# # cv2.destroyAllWindows()


# # 1st code 
 
# import cv2
# import dlib
# import numpy as np
# import time

# # Load the pre-trained face detector and facial landmarks predictor from dlib
# detector = dlib.get_frontal_face_detector()

# # Download this model from dlib's website
# predictor = dlib.shape_predictor("C:/Users/Aakash/Downloads/final project/Proctoring-AI-master/shape_predictor_68_face_landmarks.dat")

# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     start = time.time()

#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = detector(gray)

#     for face in faces:
#         # Get the landmarks/parts for the face in box
#         landmarks = predictor(gray, face)

#         # Points for head pose estimation
#         image_points = np.array([
#             (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
#             (landmarks.part(8).x, landmarks.part(8).y),    # Chin
#             (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
#             (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
#             (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
#             (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
#         ], dtype=np.float64)

#         # 3D model points.
#         model_points = np.array([
#             (0.0, 0.0, 0.0),             # Nose tip
#             (0.0, -330.0, -65.0),        # Chin
#             (-225.0, 170.0, -135.0),     # Left eye left corner
#             (225.0, 170.0, -135.0),      # Right eye right corner
#             (-150.0, -150.0, -125.0),    # Left Mouth corner
#             (150.0, -150.0, -125.0)      # Right mouth corner
#         ])

#         # Camera matrix
#         focal_length = image.shape[1]
#         cam_matrix = np.array([
#             [focal_length, 0, image.shape[1] / 2],
#             [0, focal_length, image.shape[0] / 2],
#             [0, 0, 1]
#         ], dtype=np.float64)

#         dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

#         # SolvePnP
#         success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

#         # Get rotation matrix and angles
#         rmat, _ = cv2.Rodrigues(rot_vec)
#         angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
#         # Extract the angles from the output
#         angle_x, angle_y, angle_z = angles  # angles contains rotation angles for X, Y, Z axes

#         # X, y, Z = angles * 360
        
#         # Convert radians to degrees
#         X, y, Z = angle_x * 360, angle_y * 360, angle_z * 360

#         # Determine head pose
#         if y < -10:
#             text = "Looking Left"
#         elif y > 10:
#             text = "Looking Right"
#         elif X < -10:
#             text = "Looking Down"
#         elif X > 10:
#             text = "Looking Up"
#         else:
#             text = "Forward"

#         # Draw the landmarks and pose direction
#         for (x, y) in image_points:
#             cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

#         cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
#         cv2.putText(image, f"X: {X:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, f"Y: {y:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, f"Z: {Z:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     end = time.time()
#     fps = 1 / (end - start)
#     cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

#     cv2.imshow('Head Pose Estimation', image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import dlib
import numpy as np
import time

# Load the pre-trained face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/Aakash/Downloads/final project/Proctoring-AI-master/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    start = time.time()

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    for face in faces:
        # Get the landmarks/parts for the face in box
        landmarks = predictor(gray, face)

        # Points for head pose estimation
        image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),    # Chin
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
        ], dtype=np.float64)

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Camera matrix
        focal_length = image.shape[1]
        cam_matrix = np.array([
            [focal_length, 0, image.shape[1] / 2],
            [0, focal_length, image.shape[0] / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # SolvePnP
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

        # Get rotation matrix and angles
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Extract the angles from the output
        angle_x, angle_y, angle_z = angles  # angles contains rotation angles for X, Y, Z axes

        # Convert radians to degrees
        X, Y, Z = angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi

        # Determine head pose
        if Y < -15:
            text = "Looking Left"
        elif Y > 15:
            text = "Looking Right"
        elif X < -10:
            text = "Looking Down"
        elif X > 10:
            text = "Looking Up"
        else:
            text = "Forward"

        # Draw the landmarks and pose direction
        for (x, y) in image_points:
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(image, f"X: {X:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f"Y: {Y:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f"Z: {Z:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    end = time.time()
    fps = 1 / (end - start)
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
