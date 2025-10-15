import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
color_img = cv2.imread('images/atmwithdrawal.png')
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# Step 1: Thresholding (Inverted to detect shapes)
_, binary_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

# Step 2: Find contours
contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step 3: Filter contours (remove lines/arrows)
filtered_contours = contours 
# for c in contours:
#     area = cv2.contourArea(c)
#     x, y, w, h = cv2.boundingRect(c)
#     aspect_ratio = max(w, h) / min(w, h)
#     # if area > 300 and aspect_ratio < 5:
#     filtered_contours.append(c)
            
print(f"Number of contours found: {len(filtered_contours)}")
# Step 4: Draw bounding boxes on original image
bounding_box_img = color_img.copy()
for c in filtered_contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(bounding_box_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert BGR to RGB for matplotlib
original_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB)  # Grayscale to RGB for display
contour_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)  # Draw all contours in red
final_rgb = cv2.cvtColor(bounding_box_img, cv2.COLOR_BGR2RGB)

# Create subplots
plt.figure(figsize=(15, 10))

# Subplot 1: Original Image
plt.subplot(2, 2, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('on')

# Subplot 2: Binary Thresholded Image
plt.subplot(2, 2, 2)
plt.imshow(binary_img, cmap='gray')
plt.title('Binary Image (Inverted)')
plt.axis('on')

# Subplot 3: All Detected Contours (Before Filtering)
plt.subplot(2, 2, 3)
plt.imshow(contour_img)
plt.title(f'All Contours ({len(contours)} detected)')
plt.axis('on')

# Subplot 4: Final Bounding Boxes (After Filtering)
plt.subplot(2, 2, 4)
plt.imshow(final_rgb)
plt.title(f'Filtered Bounding Boxes ({len(filtered_contours)} kept)')
plt.axis('on')

plt.tight_layout()
plt.show()