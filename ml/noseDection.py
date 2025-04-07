import cv2
import numpy as np

# Load the image
image_path = 'face.jpg'  # 🔁 Replace with your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')  # Make sure the file is present in your script directory

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    # Draw face rectangle
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]

    # Detect eyes in face region
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=3)
    for (ex, ey, ew, eh) in eyes:
        center = (ex + ew // 2, ey + eh // 2)
        radius = ew // 4
        cv2.circle(roi_color, center, radius, (0, 255, 0), 2)

        # Define eye ROI for pupil detection
        eye_gray = roi_gray[ey:ey + eh, ex:ex + ew]
        eye_eq = cv2.equalizeHist(eye_gray)
        eye_blur = cv2.GaussianBlur(eye_eq, (9, 9), 2)

        # Detect pupils using HoughCircles
        circles = cv2.HoughCircles(eye_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                   param1=100, param2=15, minRadius=3, maxRadius=10)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (cx, cy, r) in circles[0, :]:
                cv2.circle(roi_color[ey:ey + eh, ex:ex + ew], (cx, cy), r, (0, 0, 255), 2)

    # Detect nose
    nose = nose_cascade.detectMultiScale(roi_gray, 1.1, 5)
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)
        break  # Only one nose

# Show result
cv2.imshow('Face, Eyes, Nose, and Pupil Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
