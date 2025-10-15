import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
color_img = cv2.imread('images/atmwithdrawal.png')
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# Binarize the image (black writing -> white for contour detection)
_, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter small contours (adjust area threshold as needed)
min_area = 100
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]

print(f"Number of contours found: {len(filtered_contours)}")

# Draw bounding boxes and contours
output_img = color_img.copy()
for i, c in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
    cv2.drawContours(output_img, [c], -1, (255, 0, 0), 1)  # Blue contour
    cv2.putText(output_img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Label

# Convert BGR to RGB for matplotlib
output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

# Display
plt.figure(figsize=(10, 10))
plt.imshow(output_img_rgb)
plt.title(f"Detected Contours (n={len(filtered_contours)})")
plt.axis('off')
plt.show()