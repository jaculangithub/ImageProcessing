import cv2
import numpy as np

# Load the image
img = cv2.imread('images/login.png')  # Replace with your image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 1: Edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Step 2: Hough Line Transform
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=100,
    minLineLength=50,
    maxLineGap=10
)

# Step 3: Draw lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Step 4: Show the result
cv2.imshow('Detected Lines', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
