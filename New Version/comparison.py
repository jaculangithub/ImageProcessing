import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("./images/login.png")  # replace with your image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Threshold to get binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# 2. Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Make copies for drawing
binary_result = image.copy()
canny_result = image.copy()

# --- Hough Line on binary image ---
lines_binary = cv2.HoughLines(binary, 1, np.pi/180, 2)
if lines_binary is not None:
    for rho, theta in lines_binary[:,0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(binary_result, (x1,y1), (x2,y2), (0,0,255), 2)

# --- Hough Line on Canny edges ---
lines_canny = cv2.HoughLines(edges, 1, np.pi/180, 2)
if lines_canny is not None:
    for rho, theta in lines_canny[:,0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(canny_result, (x1,y1), (x2,y2), (0,255,0), 2)

# Convert to RGB for Matplotlib
binary_result_rgb = cv2.cvtColor(binary_result, cv2.COLOR_BGR2RGB)
canny_result_rgb = cv2.cvtColor(canny_result, cv2.COLOR_BGR2RGB)
binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.imshow(binary_rgb)
plt.title("Binary Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(binary_result_rgb)
plt.title("Hough on Binary")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(edges_rgb)
plt.title("Canny Edges")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(canny_result_rgb)
plt.title("Hough on Canny")
plt.axis("off")

plt.show()

