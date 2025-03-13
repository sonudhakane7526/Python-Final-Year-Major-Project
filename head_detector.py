import cv2
import numpy as np

def get_head_cascade():
    head_cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')  # Make sure this file is in the same directory as your script
    return head_cascade

def find_heads(img, model):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    heads = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return heads

def draw_heads(img, heads):
    for (x, y, w, h) in heads:
        # Calculate the center and radius for the circle
        center_x = x + w // 2
        center_y = y + h // 2
        radius = int((w + h) // 8)  # Adjust radius as needed
        
        # Draw the circle
        cv2.circle(img, (center_x, center_y), radius, (0, 255, 0), 3)

def main():
    # Initialize the head detector
    head_model = get_head_cascade()

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

        heads = find_heads(frame, head_model)
        draw_heads(frame, heads)

        # Display the output
        cv2.imshow('Head Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
