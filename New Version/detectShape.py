import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread("./images/atmwithdrawal.png")
output = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reduce noise
gray = cv2.medianBlur(gray, 5)

# Detect circles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=5,      
    param1=100,         
    param2=30,       
    minRadius=10,       
    maxRadius=80
)

# Draw only the first detected circle
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, r = circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  # Circle outline
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point


# Convert BGR to RGB for matplotlib
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Show result using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(output_rgb)
plt.axis('off')
plt.title('Detected Circle')
plt.show()