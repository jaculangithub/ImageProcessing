import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
image = cv2.imread("./images/atmwithdrawal.png")
output = image.copy()

# Step 1: Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Threshold to get binary image
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Step 4: Find contours with hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

min_area = 200
contourlength = 0

# Ensure hierarchy is not None
if hierarchy is not None:
    hierarchy = hierarchy[0]  # Flatten

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        # Skip small contours
        if area <= min_area:
            continue

        parent_idx = hierarchy[i][3]  # index of parent contour
        keep = True
        
        # If contour has a parent, check relative area
        if parent_idx != -1:
            parent_area = cv2.contourArea(contours[parent_idx])
            if area > 0.85 * parent_area:  # child is too big ( >85% of parent )
                keep = False

        if keep:
            contourlength += 1
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Draw red dots on corners
            corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
            for corner in corners:
                cv2.circle(output, corner, 2, (0, 0, 255), -1)

# Step 5: Detect lines using Hough Line Transform
# Create a copy for line detection
line_image = image.copy()

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect lines using HoughLinesP (Probabilistic Hough Transform)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=2)

# Draw detected lines in blue
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line angle to classify as horizontal, vertical, or diagonal
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        
        # Draw all detected lines in blue
        cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue lines on separate image

# Convert images for matplotlib
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

print(f"Total filtered contours with area > {min_area}: {contourlength}")
print(f"Detected {len(lines) if lines is not None else 0} lines")

# Show all processing steps including line detection
plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(edges_rgb, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(thresh_rgb, cmap='gray')
plt.title("Thresholded Image")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(line_image_rgb)
plt.title("Detected Lines Only (Blue)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(output_rgb)
plt.title(f"Contours + Lines\n({contourlength} contours, {len(lines) if lines is not None else 0} lines)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(gray_rgb, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()