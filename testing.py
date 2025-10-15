import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pytesseract

# Read image using PIL for pytesseract
pil_img = Image.open('images/atmwithdrawalv2.png')
color_img = cv2.imread('images/atmwithdrawalv2.png')
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# Step 1: Use pytesseract to detect text and get bounding boxes
data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)

# Create a copy of the image to modify (remove text areas)
modified_img = color_img.copy()

# Remove text areas by changing them to background color
for i in range(len(data['text'])):
    # Skip empty text
    if not data['text'][i].strip():
        continue
        
    # Get the bounding box coordinates
    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
    
    # Get the average background color around the text area
    # Expand the area slightly to sample background
    bg_sample_x1 = max(0, x - 5)
    bg_sample_y1 = max(0, y - 5)
    bg_sample_x2 = min(color_img.shape[1], x + w + 5)
    bg_sample_y2 = min(color_img.shape[0], y + h + 5)
    
    # Sample background color from the expanded area
    bg_area = color_img[bg_sample_y1:bg_sample_y2, bg_sample_x1:bg_sample_x2]
    if bg_area.size > 0:
        bg_color = np.median(bg_area, axis=(0, 1)).astype(int)
    else:
        bg_color = [255, 255, 255]  # Default white background
    
    # Fill the text area with background color
    cv2.rectangle(modified_img, (x, y), (x + w, y + h), bg_color.tolist(), -1)
    
    print(f"Text: '{data['text'][i]}', Bounding Box: (x={x}, y={y}, w={w}, h={h})")

# Convert modified image to grayscale for contour detection
modified_gray = cv2.cvtColor(modified_img, cv2.COLOR_BGR2GRAY)

# Step 2: Thresholding (Inverted to detect shapes) on modified image
_, binary_img = cv2.threshold(modified_gray, 240, 255, cv2.THRESH_BINARY_INV)

# Step 3: Find contours on the modified image
contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Step 4: Filter contours (remove lines/arrows if needed)
filtered_contours = []
for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
    
    # Basic filtering - adjust thresholds as needed
    if area > 100 and aspect_ratio < 10:  # Remove very small and very elongated contours
        filtered_contours.append(c)

print(f"Number of contours found: {len(contours)}")
print(f"Number of contours after filtering: {len(filtered_contours)}")

# Step 5: Draw bounding boxes on original image
bounding_box_img = color_img.copy()
for c in filtered_contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(bounding_box_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Convert BGR to RGB for matplotlib
original_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
modified_rgb = cv2.cvtColor(modified_img, cv2.COLOR_BGR2RGB)
binary_rgb = cv2.cvtColor(binary_img, cv2.COLOR_BGR2RGB)
contour_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
final_rgb = cv2.cvtColor(bounding_box_img, cv2.COLOR_BGR2RGB)

# Create subplots
plt.figure(figsize=(20, 12))

# Subplot 1: Original Image
plt.subplot(2, 3, 1)
plt.imshow(original_rgb)
plt.title('Original Image')
plt.axis('on')

# Subplot 2: Image with Text Removed
plt.subplot(2, 3, 2)
plt.imshow(modified_rgb)
plt.title('Image with Text Areas Removed')
plt.axis('on')

# Subplot 3: Binary Thresholded Image
plt.subplot(2, 3, 3)
plt.imshow(binary_img, cmap='gray')
plt.title('Binary Image (Inverted)')
plt.axis('on')

# Subplot 4: All Detected Contours (Before Filtering)
plt.subplot(2, 3, 4)
plt.imshow(contour_img)
plt.title(f'All Contours ({len(contours)} detected)')
plt.axis('on')

# Subplot 5: Final Bounding Boxes (After Filtering)
plt.subplot(2, 3, 5)
plt.imshow(final_rgb)
plt.title(f'Filtered Bounding Boxes ({len(filtered_contours)} kept)')
plt.axis('on')

plt.tight_layout()
plt.show()