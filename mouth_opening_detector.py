import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

face_model = get_face_detector()
landmark_model = get_landmark_model()
outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0] * 5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0] * 3
font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        continue
    rects = find_faces(img, face_model)
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        draw_marks(img, shape)
        cv2.putText(img, 'Press r to record Mouth distances', (30, 30), font,
                    1, (0, 255, 255), 2)
        cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        for _ in range(100):
            ret, img = cap.read()
            if not ret:
                continue
            rects = find_faces(img, face_model)
            for rect in rects:
                shape = detect_marks(img, landmark_model, rect)
                for j, (p1, p2) in enumerate(outer_points):
                    d_outer[j] += shape[p2][1] - shape[p1][1]
                for j, (p1, p2) in enumerate(inner_points):
                    d_inner[j] += shape[p2][1] - shape[p1][1]
        break
cv2.destroyAllWindows()
d_outer[:] = [x / 100 for x in d_outer]
d_inner[:] = [x / 100 for x in d_inner]

while True:
    ret, img = cap.read()
    if not ret:
        break
    rects = find_faces(img, face_model)
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        cnt_outer = 0
        cnt_inner = 0
        draw_marks(img, shape[48:])
        for j, (p1, p2) in enumerate(outer_points):
            if d_outer[j] + 3 < shape[p2][1] - shape[p1][1]:
                cnt_outer += 1
        for j, (p1, p2) in enumerate(inner_points):
            if d_inner[j] + 2 < shape[p2][1] - shape[p1][1]:
                cnt_inner += 1
        if cnt_outer > 3 and cnt_inner > 1:
            message = 'Mouth open'
        else:
            message = 'Normal'
        # cv2.putText(img, message, (30, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(img, message, (30, 30), font, 1.5, (0, 0, 255), 3)
    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()













































































# while True:
#     ret, img = cap.read()
#     if not ret:
#         continue
#     rects = find_faces(img, face_model)
#     for rect in rects:
#         shape = detect_marks(img, landmark_model, rect)
#         cnt_outer = 0
#         cnt_inner = 0
#         draw_marks(img, shape[48:])
#         for j, (p1, p2) in enumerate(outer_points):
#             if d_outer[j] + 3 < shape[p2][1] - shape[p1][1]:
#                 cnt_outer += 1
#         for j, (p1, p2) in enumerate(inner_points):
#             if d_inner[j] + 2 < shape[p2][1] - shape[p1][1]:
#                 cnt_inner += 1
#         if cnt_outer > 3 and cnt_inner > 1:
#             print('Mouth open')
#         else:
#             message = 'Normal'
#             cv2.putText(img, 'Mouth open', 'Normal', (30, 30), font,
#                         2, (0, 0, 255), 2.5)
#         # show the output image with the face detections + facial landmarks
#     cv2.imshow("Output", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
