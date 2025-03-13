# import cv2
# import dlib
# import numpy as np
# import time

# # Load the pre-trained face detector and facial landmarks predictor from dlib
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/Aakash/Downloads/final project/Proctoring-AI-master/shape_predictor_68_face_landmarks.dat")

# # The actual path to your video file
# video_path = 'C:\\Users\\Aakash\\Downloads\\final project\\Proctoring-AI-master\\eye_tracking\\video.mp4'
# cap = cv2.VideoCapture(video_path)

# # Check if the video capture is initialized
# if not cap.isOpened():
#     print(f"Error: Could not open video {video_path}.")
#     exit()

# while cap.isOpened():
#     # Capture frame-by-frame
#     success, image = cap.read()
#     if not success:
#         print("End of video or error reading frame.")
#         break

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
#             (landmarks.part(48).x, landmarks.part(48).y),  # Left mouth corner
#             (landmarks.part(54).x, landmarks.part(54).y)   # Right mouth corner
#         ], dtype=np.float64)

#         # 3D model points.
#         model_points = np.array([
#             (0.0, 0.0, 0.0),             # Nose tip
#             (0.0, -330.0, -65.0),        # Chin
#             (-225.0, 170.0, -135.0),     # Left eye left corner
#             (225.0, 170.0, -135.0),      # Right eye right corner
#             (-150.0, -150.0, -125.0),    # Left mouth corner
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

#         # SolvePnP to obtain rotation and translation vectors
#         success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

#         # Get rotation matrix and angles
#         rmat, _ = cv2.Rodrigues(rot_vec)
#         angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

#         # Extract the angles from the output
#         angle_x, angle_y, angle_z = angles  # angles contains rotation angles for X, Y, Z axes

#         # Convert radians to degrees
#         X, Y, Z = np.degrees([angle_x, angle_y, angle_z])

#         # Determine head pose
#         if Y < -15:
#             text = "Looking Left"
#         elif Y > 15:
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

#         # Display pose information on the image
#         cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
#         cv2.putText(image, f"X: {X:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, f"Y: {Y:.2f}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(image, f"Z: {Z:.2f}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Calculate and display FPS
#     end = time.time()
#     fps = 1 / (end - start)
#     cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

#     # Show the result
#     cv2.imshow('Head Pose Estimation', image)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import math

# def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
#     point_3d = []
#     dist_coeffs = np.zeros((4, 1))
#     rear_size = val[0]
#     rear_depth = val[1]
#     point_3d.append((-rear_size, -rear_size, rear_depth))
#     point_3d.append((-rear_size, rear_size, rear_depth))
#     point_3d.append((rear_size, rear_size, rear_depth))
#     point_3d.append((rear_size, -rear_size, rear_depth))
#     point_3d.append((-rear_size, -rear_size, rear_depth))

#     front_size = val[2]
#     front_depth = val[3]
#     point_3d.append((-front_size, -front_size, front_depth))
#     point_3d.append((-front_size, front_size, front_depth))
#     point_3d.append((front_size, front_size, front_depth))
#     point_3d.append((front_size, -front_size, front_depth))
#     point_3d.append((-front_size, -front_size, front_depth))
#     point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

#     (point_2d, _) = cv2.projectPoints(point_3d,
#                                       rotation_vector,
#                                       translation_vector,
#                                       camera_matrix,
#                                       dist_coeffs)
#     point_2d = np.int32(point_2d.reshape(-1, 2))
#     return point_2d

# def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
#                         rear_size=300, rear_depth=0, front_size=500, front_depth=400,
#                         color=(255, 255, 0), line_width=2):
#     rear_size = 1
#     rear_depth = 0
#     front_size = img.shape[1]
#     front_depth = front_size * 2
#     val = [rear_size, rear_depth, front_size, front_depth]
#     point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
#     cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
#     cv2.line(img, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
#     cv2.line(img, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
#     cv2.line(img, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)

# def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
#     rear_size = 1
#     rear_depth = 0
#     front_size = img.shape[1]
#     front_depth = front_size * 2
#     val = [rear_size, rear_depth, front_size, front_depth]
#     point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
#     y = (point_2d[5] + point_2d[8]) // 2
#     x = point_2d[2]
#     return (x, y)

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# size = img.shape
# font = cv2.FONT_HERSHEY_SIMPLEX

# # Define model points
# model_points = np.array([
#     (0.0, 0.0, 0.0),  # Nose tip
#     (0.0, -330.0, -65.0),  # Chin
#     (-225.0, 170.0, -135.0),  # Left eye left corner
#     (225.0, 170.0, -135.0),  # Right eye right corner
#     (-150.0, -150.0, -125.0),  # Left mouth corner
#     (150.0, -150.0, -125.0)  # Right mouth corner
# ])

# focal_length = size[1]
# center = (size[1] / 2, size[0] / 2)
# camera_matrix = np.array(
#     [[focal_length, 0, center[0]],
#      [0, focal_length, center[1]],
#      [0, 0, 1]], dtype="double"
# )

# while True:
#     ret, img = cap.read()
#     if ret:
#         # Generate fake landmarks or set them to predefined values for the purpose of testing
#         fake_marks = np.array([
#             (img.shape[1] // 2, img.shape[0] // 2),  # Fake nose tip
#             (img.shape[1] // 2, img.shape[0] // 2 + 100),  # Fake chin
#             (img.shape[1] // 2 - 50, img.shape[0] // 2 - 50),  # Fake left eye
#             (img.shape[1] // 2 + 50, img.shape[0] // 2 - 50),  # Fake right eye
#             (img.shape[1] // 2 - 50, img.shape[0] // 2 + 50),  # Fake left mouth
#             (img.shape[1] // 2 + 50, img.shape[0] // 2 + 50)  # Fake right mouth
#         ], dtype="double")

#         dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
#         (success, rotation_vector, translation_vector) = cv2.solvePnP(
#             model_points, fake_marks, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP
#         )

#         (nose_end_point2D, _) = cv2.projectPoints(
#             np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
#         )

#         for p in fake_marks:
#             cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

#         p1 = (int(fake_marks[0][0]), int(fake_marks[0][1]))
#         p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
#         x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

#         cv2.line(img, p1, p2, (0, 255, 255), 2)
#         cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)

#         try:
#             m = (p2[1] - p1[1]) / (p2[0] - p1[0])
#             ang1 = int(math.degrees(math.atan(m)))
#         except ZeroDivisionError:
#             ang1 = 90

#         try:
#             m = (x2[1] - x1[1]) / (x2[0] - x1[0])
#             ang2 = int(math.degrees(math.atan(-1 / m)))
#         except ZeroDivisionError:
#             ang2 = 90

#         if ang1 >= 48:
#             print('Head down')
#             cv2.putText(img, 'Head down', (30, 30), font, 2, (255, 255, 128), 3)
#         elif ang1 <= -48:
#             print('Head up')
#             cv2.putText(img, 'Head up', (30, 30), font, 2, (255, 255, 128), 3)

#         if ang2 >= 48:
#             print('Head right')
#             cv2.putText(img, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
#         elif ang2 <= -48:
#             print('Head left')
#             cv2.putText(img, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)

#         cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
#         cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
#         cv2.imshow('img', img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# cv2.destroyAllWindows()
# cap.release()

import cv2
import dlib
import numpy as np
import time
import math

def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4, 1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)

def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size * 2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8]) // 2
    x = point_2d[2]
    return (x, y)

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

        if success:
            # Draw the annotation box
            draw_annotation_box(image, rot_vec, trans_vec, cam_matrix)

            # Get 2D points for head pose arrows
            x1, x2 = head_pose_points(image, rot_vec, trans_vec, cam_matrix)

            # Project the nose end point
            (nose_end_point2D, _) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_matrix, dist_coeffs
            )

            p1 = (int(image_points[0][0]), int(image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(image, p1, p2, (0, 255, 255), 2)
            cv2.line(image, tuple(x1), tuple(x2), (255, 255, 0), 2)

            # Compute angles
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            angle_x, angle_y, angle_z = angles  # angles contains rotation angles for X, Y, Z axes

            # Convert radians to degrees
            X, Y, Z = angle_x * 180 / np.pi, angle_y * 180 / np.pi, angle_z * 180 / np.pi

            if Y < -20:
                text = "Looking Left"
            elif Y > 20:
                text = "Looking Right"
            elif X < -15:
                text = "Looking Down"
            elif X > 15:
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

