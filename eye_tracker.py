# import cv2
# import numpy as np
# from face_detector import get_face_detector, find_faces
# from face_landmarks import get_landmark_model, detect_marks


# def eye_on_mask(mask, side, shape):
#     points = [shape[i] for i in side]
#     points = np.array(points, dtype=np.int32)
#     mask = cv2.fillConvexPoly(mask, points, 255)
#     l = points[0][0]
#     t = (points[1][1]+points[2][1])//2
#     r = points[3][0]
#     b = (points[4][1]+points[5][1])//2
#     return mask, [l, t, r, b]
# def find_eyeball_position(end_points, cx, cy):
#     x_ratio = (end_points[0] - cx)/(cx - end_points[2])
#     y_ratio = (cy - end_points[1])/(end_points[3] - cy)
#     if x_ratio > 3:
#         return 1
#     elif x_ratio < 0.33:
#         return 2
#     elif y_ratio < 0.33:
#         return 3
#     else:
#         return 0

    
# def contouring(thresh, mid, img, end_points, right=False):
#     cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     try:
#         cnt = max(cnts, key = cv2.contourArea)
#         M = cv2.moments(cnt)
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
#         if right:
#             cx += mid
#         cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
#         pos = find_eyeball_position(end_points, cx, cy)
#         return pos
#     except:
#         pass
    
# def process_thresh(thresh):
#     thresh = cv2.erode(thresh, None, iterations=2) 
#     thresh = cv2.dilate(thresh, None, iterations=4) 
#     thresh = cv2.medianBlur(thresh, 3) 
#     thresh = cv2.bitwise_not(thresh)
#     return thresh

# def print_eye_pos(img, left, right):
#     if left == right and left != 0:
#         text = ''
#         if left == 1:
#             print('Looking left')
#             text = 'Looking left'
#         elif left == 2:
#             print('Looking right')
#             text = 'Looking right'
#         elif left == 3:
#             print('Looking up')
#             text = 'Looking up'
#         font = cv2.FONT_HERSHEY_SIMPLEX 
#         cv2.putText(img, text, (30, 30), font,  
#                    1, (0, 255, 255), 2, cv2.LINE_AA) 

# face_model = get_face_detector()
# landmark_model = get_landmark_model()
# left = [36, 37, 38, 39, 40, 41]
# right = [42, 43, 44, 45, 46, 47]

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# thresh = img.copy()

# cv2.namedWindow('image')
# kernel = np.ones((9, 9), np.uint8)

# def nothing(x):
#     pass
# cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

# while(True):
#     ret, img = cap.read()
#     rects = find_faces(img, face_model)
    
#     for rect in rects:
#         shape = detect_marks(img, landmark_model, rect)
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         mask, end_points_left = eye_on_mask(mask, left, shape)
#         mask, end_points_right = eye_on_mask(mask, right, shape)
#         mask = cv2.dilate(mask, kernel, 5)
        
#         eyes = cv2.bitwise_and(img, img, mask=mask)
#         mask = (eyes == [0, 0, 0]).all(axis=2)
#         eyes[mask] = [255, 255, 255]
#         mid = int((shape[42][0] + shape[39][0]) // 2)
#         eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
#         threshold = cv2.getTrackbarPos('threshold', 'image')
#         _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
#         thresh = process_thresh(thresh)
        
#         eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
#         eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
#         print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
#         # for (x, y) in shape[36:48]:
#         #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        
#     cv2.imshow('eyes', img)
#     cv2.imshow("image", thresh)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from face_detector import get_face_detector, find_faces
# from face_landmarks import get_landmark_model, detect_marks


# def eye_on_mask(mask, side, shape):
#     points = [shape[i] for i in side]
#     points = np.array(points, dtype=np.int32)
#     mask = cv2.fillConvexPoly(mask, points, 255)
#     l = points[0][0]
#     t = (points[1][1]+points[2][1])//2
#     r = points[3][0]
#     b = (points[4][1]+points[5][1])//2
#     return mask, [l, t, r, b]

# def find_eyeball_position(end_points, cx, cy):
#     x_ratio = (end_points[0] - cx)/(cx - end_points[2])
#     y_ratio = (cy - end_points[1])/(end_points[3] - cy)
#     if x_ratio > 3:
#         return 1
#     elif x_ratio < 0.33:
#         return 2
#     elif y_ratio < 0.33:
#         return 3
#     else:
#         return 0

    
# def contouring(thresh, mid, img, end_points, right=False):
#     cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     try:
#         cnt = max(cnts, key = cv2.contourArea)
#         M = cv2.moments(cnt)
#         cx = int(M['m10']/M['m00'])
#         cy = int(M['m01']/M['m00'])
#         if right:
#             cx += mid
#         cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
#         pos = find_eyeball_position(end_points, cx, cy)
#         return pos
#     except:
#         pass
    
# def process_thresh(thresh):
#     thresh = cv2.erode(thresh, None, iterations=2) 
#     thresh = cv2.dilate(thresh, None, iterations=4) 
#     thresh = cv2.medianBlur(thresh, 3) 
#     thresh = cv2.bitwise_not(thresh)
#     return thresh

# def print_eye_pos(img, left, right):
#     if left == right and left != 0:
#         text = ''
#         if left == 1:
#             print('Looking left')
#             text = 'Looking left'
#         elif left == 2:
#             print('Looking right')
#             text = 'Looking right'
#         elif left == 3:
#             print('Looking up')
#             text = 'Looking up'
#         font = cv2.FONT_HERSHEY_SIMPLEX 
#         cv2.putText(img, text, (30, 30), font,  
#                    1, (0, 255, 255), 2, cv2.LINE_AA) 

# face_model = get_face_detector()
# landmark_model = get_landmark_model()
# left = [36, 37, 38, 39, 40, 41]
# right = [42, 43, 44, 45, 46, 47]

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# thresh = img.copy()

# cv2.namedWindow('image')
# kernel = np.ones((9, 9), np.uint8)

# def nothing(x):
#     pass
# cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

# while(True):
#     ret, img = cap.read()
#     rects = find_faces(img, face_model)
    
#     for rect in rects:
#         shape = detect_marks(img, landmark_model, rect)
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         mask, end_points_left = eye_on_mask(mask, left, shape)
#         mask, end_points_right = eye_on_mask(mask, right, shape)
#         mask = cv2.dilate(mask, kernel, 5)
        
#         eyes = cv2.bitwise_and(img, img, mask=mask)
#         mask = (eyes == [0, 0, 0]).all(axis=2)
#         eyes[mask] = [255, 255, 255]
#         mid = int((shape[42][0] + shape[39][0]) // 2)
#         eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
#         threshold = cv2.getTrackbarPos('threshold', 'image')
#         _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
#         thresh = process_thresh(thresh)
        
#         eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
#         eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
#         print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
#         # for (x, y) in shape[36:48]:
#         #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
        
#     cv2.imshow('eyes', img)
#     cv2.imshow("image", thresh)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# from face_detector import get_face_detector, find_faces
# from face_landmarks import get_landmark_model, detect_marks

# def eye_on_mask(mask, side, shape):
#     points = [shape[i] for i in side]
#     points = np.array(points, dtype=np.int32)
#     mask = cv2.fillConvexPoly(mask, points, 255)
#     l = points[0][0]
#     t = (points[1][1] + points[2][1]) // 2
#     r = points[3][0]
#     b = (points[4][1] + points[5][1]) // 2
#     return mask, [l, t, r, b]

# def find_eyeball_position(end_points, cx, cy):
#     x_ratio = (end_points[0] - cx) / (cx - end_points[2])
#     y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
#     if x_ratio > 3:
#         return 1  # Looking left
#     elif x_ratio < 0.33:
#         return 2  # Looking right
#     elif y_ratio < 0.33:
#         return 3  # Looking up
#     else:
#         return 0  # Looking forward

# def contouring(thresh, mid, img, end_points, right=False):
#     cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     try:
#         cnt = max(cnts, key=cv2.contourArea)
#         M = cv2.moments(cnt)
#         cx = int(M['m10'] / M['m00'])
#         cy = int(M['m01'] / M['m00'])
#         if right:
#             cx += mid
#         cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
#         pos = find_eyeball_position(end_points, cx, cy)
#         return pos
#     except:
#         return 0

# def process_thresh(thresh):
#     thresh = cv2.erode(thresh, None, iterations=2)
#     thresh = cv2.dilate(thresh, None, iterations=4)
#     thresh = cv2.medianBlur(thresh, 3)
#     thresh = cv2.bitwise_not(thresh)
#     return thresh

# def print_eye_pos(img, left, right):
#     h, w, _ = img.shape  # Get the dimensions of the image
#     color = (0, 255, 255)  # Yellow color for the lines
#     thickness = 2  # Thickness of the lines

#     if left == right and left != 0:
#         if left == 1:  # Looking left
#             print('Looking left')
#             cv2.line(img, (w // 4, h // 2), (w // 2, h // 2), color, thickness)
#         elif left == 2:  # Looking right
#             print('Looking right')
#             cv2.line(img, (w // 2, h // 2), (3 * w // 4, h // 2), color, thickness)
#         elif left == 3:  # Looking up
#             print('Looking up')
#             cv2.line(img, (w // 2, h // 4), (w // 2, h // 2), color, thickness)
#         else:  # Looking forward
#             print('Looking forward')
#             cv2.line(img, (w // 3, h // 2), (2 * w // 3, h // 2), color, thickness)
#             cv2.line(img, (w // 2, h // 3), (w // 2, 2 * h // 3), color, thickness)

# # Assuming get_face_detector, get_landmark_model, find_faces, detect_marks are defined elsewhere
# face_model = get_face_detector()
# landmark_model = get_landmark_model()
# left = [36, 37, 38, 39, 40, 41]
# right = [42, 43, 44, 45, 46, 47]

# cap = cv2.VideoCapture(0)
# ret, img = cap.read()
# thresh = img.copy()

# cv2.namedWindow('image')
# kernel = np.ones((9, 9), np.uint8)

# def nothing(x):
#     pass

# cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

# while True:
#     ret, img = cap.read()
#     rects = find_faces(img, face_model)
    
#     for rect in rects:
#         shape = detect_marks(img, landmark_model, rect)
#         mask = np.zeros(img.shape[:2], dtype=np.uint8)
#         mask, end_points_left = eye_on_mask(mask, left, shape)
#         mask, end_points_right = eye_on_mask(mask, right, shape)
#         mask = cv2.dilate(mask, kernel, 5)
        
#         eyes = cv2.bitwise_and(img, img, mask=mask)
#         mask = (eyes == [0, 0, 0]).all(axis=2)
#         eyes[mask] = [255, 255, 255]
#         mid = int((shape[42][0] + shape[39][0]) // 2)
#         eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
#         threshold = cv2.getTrackbarPos('threshold', 'image')
#         _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
#         thresh = process_thresh(thresh)
        
#         eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
#         eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
#         print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
        
#     cv2.imshow('eyes', img)
#     cv2.imshow("image", thresh)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    x_ratio = (end_points[0] - cx) / (cx - end_points[2])
    y_ratio = (cy - end_points[1]) / (end_points[3] - cy)
    if x_ratio > 3:
        return 1  # Looking left
    elif x_ratio < 0.33:
        return 2  # Looking right
    elif y_ratio < 0.33:
        return 3  # Looking up
    else:
        return 0  # Looking forward

def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        return 0

def process_thresh(thresh):
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):
    h, w, _ = img.shape  # Get the dimensions of the image
    color = (0, 0, 255)  # Red color for text
    thickness = 2  # Thickness of the text
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the text
    text = ""

    if left == right and left != 0:
        if left == 1:  # Looking left
            text = 'Looking left'
        elif left == 2:  # Looking right
            text = 'Looking right'
        elif left == 3:  # Looking up
            text = 'Looking up'
        else:  # Looking forward
            text = 'Looking forward'

    # Display the text on the camera feed
    if text:
        cv2.putText(img, text, (30, 30), font, 1, color, thickness, cv2.LINE_AA)

# Assuming get_face_detector, get_landmark_model, find_faces, detect_marks are defined elsewhere
face_model = get_face_detector()
landmark_model = get_landmark_model()
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass

cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

while True:
    ret, img = cap.read()
    rects = find_faces(img, face_model)
    
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)
        
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        
        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, img, end_points_right, True)
        print_eye_pos(img, eyeball_pos_left, eyeball_pos_right)
        
    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
