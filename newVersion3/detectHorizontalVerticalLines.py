import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

# Read image
image = cv2.imread('images/atmwithdrawalv2.png')
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 0: Use pytesseract to detect and remove text
def remove_text_from_image(img):
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Use pytesseract to get text bounding boxes
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Create a mask for text regions
    text_mask = np.zeros_like(img)
    
    # Process each detected text bounding box
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 30:  # Only consider confident detections
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            text = data['text'][i].strip()
            
            if text:  # Only process non-empty text
                # Get the average color of the area around the text for background inpainting
                # Expand the region slightly to get better background color
                expand = 2
                x1 = max(0, x - expand)
                y1 = max(0, y - expand)
                x2 = min(width, x + w + expand)
                y2 = min(height, y + h + expand)
                
                # Get the average background color from the expanded area
                bg_color = np.mean(img[y1:y2, x1:x2], axis=(0, 1))
                
                # Fill the text area with the background color
                cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
                
                # Also mark on the mask
                cv2.rectangle(text_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
    
    return img, text_mask

# Remove text from the image
image_no_text, text_mask = remove_text_from_image(image.copy())
gray_no_text = cv2.cvtColor(image_no_text, cv2.COLOR_BGR2GRAY)

# Step 1: Preprocessing on text-removed image
blurred = cv2.GaussianBlur(gray_no_text, (5, 5), 0)

# Step 2: Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Step 3: Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Step 4: Separate horizontal and vertical lines
horizontal_lines = []
vertical_lines = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle of the line (in degrees)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Classify as horizontal (angle near 0 or 180 degrees) or vertical (angle near 90 or -90 degrees)
        if abs(angle) < 10 or abs(angle - 180) < 10 or abs(angle + 180) < 10:
            horizontal_lines.append(line[0])
        elif abs(angle - 90) < 10 or abs(angle + 90) < 10:
            vertical_lines.append(line[0])

# Step 5: Create images for visualization
# Image with text removed (background color filled)
text_removed_img = image_no_text.copy()

# Image with all detected lines on text-removed image
all_lines_img = image_no_text.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(all_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Image with only horizontal lines (blue)
horizontal_img = image_no_text.copy()
for line in horizontal_lines:
    x1, y1, x2, y2 = line
    cv2.line(horizontal_img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue

# Image with only vertical lines (red)
vertical_img = image_no_text.copy()
for line in vertical_lines:
    x1, y1, x2, y2 = line
    cv2.line(vertical_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red

# Image with both horizontal and vertical lines
both_lines_img = image_no_text.copy()
for line in horizontal_lines:
    x1, y1, x2, y2 = line
    cv2.line(both_lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for horizontal
for line in vertical_lines:
    x1, y1, x2, y2 = line
    cv2.line(both_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for vertical

# Convert to RGB for matplotlib
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
text_removed_rgb = cv2.cvtColor(text_removed_img, cv2.COLOR_BGR2RGB)
text_mask_rgb = cv2.cvtColor(text_mask, cv2.COLOR_BGR2RGB)
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
all_lines_rgb = cv2.cvtColor(all_lines_img, cv2.COLOR_BGR2RGB)
horizontal_rgb = cv2.cvtColor(horizontal_img, cv2.COLOR_BGR2RGB)
vertical_rgb = cv2.cvtColor(vertical_img, cv2.COLOR_BGR2RGB)
both_lines_rgb = cv2.cvtColor(both_lines_img, cv2.COLOR_BGR2RGB)

# Display results
plt.figure(figsize=(20, 16))

# Original Image
plt.subplot(3, 3, 1)
plt.imshow(original_rgb)
plt.title('Original Image', fontsize=12, fontweight='bold')
plt.axis('off')

# Text Mask (where text was detected)
plt.subplot(3, 3, 2)
plt.imshow(text_mask_rgb)
plt.title('Text Detection Mask', fontsize=12, fontweight='bold')
plt.axis('off')

# Image with Text Removed
plt.subplot(3, 3, 3)
plt.imshow(text_removed_rgb)
plt.title('Image with Text Removed\n(Background color filled)', fontsize=12, fontweight='bold')
plt.axis('off')

# Edge Detection
plt.subplot(3, 3, 4)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection\n(After text removal)', fontsize=12, fontweight='bold')
plt.axis('off')

# All Detected Lines
plt.subplot(3, 3, 5)
plt.imshow(all_lines_rgb)
plt.title(f'All Lines ({len(lines) if lines is not None else 0} lines)\n(After text removal)', fontsize=12, fontweight='bold')
plt.axis('off')

# Horizontal Lines Only
plt.subplot(3, 3, 6)
plt.imshow(horizontal_rgb)
plt.title(f'Horizontal Lines ({len(horizontal_lines)} lines)', fontsize=12, fontweight='bold')
plt.axis('off')

# Vertical Lines Only
plt.subplot(3, 3, 7)
plt.imshow(vertical_rgb)
plt.title(f'Vertical Lines ({len(vertical_lines)} lines)', fontsize=12, fontweight='bold')
plt.axis('off')

# Both Horizontal and Vertical
plt.subplot(3, 3, 8)
plt.imshow(both_lines_rgb)
plt.title(f'Horizontal (Blue) & Vertical (Red) Lines\nTotal: {len(horizontal_lines) + len(vertical_lines)} lines', 
          fontsize=12, fontweight='bold')
plt.axis('off')

# Final Comparison
plt.subplot(3, 3, 9)
# Create comparison image: original on left, processed on right
comparison_img = np.hstack([original_rgb, both_lines_rgb])
plt.imshow(comparison_img)
plt.title('Comparison: Original (Left) vs Lines Detected (Right)', fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('line_detection_text_removed.png', dpi=300, bbox_inches='tight')
plt.show()

# Print results
print("=" * 60)
print("LINE DETECTION AFTER TEXT REMOVAL")
print("=" * 60)
print(f"Total lines detected: {len(lines) if lines is not None else 0}")
print(f"Horizontal lines: {len(horizontal_lines)}")
print(f"Vertical lines: {len(vertical_lines)}")
print(f"Other lines (diagonal): {len(lines) - len(horizontal_lines) - len(vertical_lines) if lines is not None else 0}")

# Show detailed information about detected lines
print("\nHORIZONTAL LINES:")
print("-" * 30)
for i, line in enumerate(horizontal_lines[:5]):  # Show first 5 as example
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(f"Line {i+1}: From ({x1}, {y1}) to ({x2}, {y2}), Length: {length:.1f} px")

print("\nVERTICAL LINES:")
print("-" * 30)
for i, line in enumerate(vertical_lines[:5]):  # Show first 5 as example
    x1, y1, x2, y2 = line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    print(f"Line {i+1}: From ({x1}, {y1}) to ({x2}, {y2}), Length: {length:.1f} px")

# Display final combined result
plt.figure(figsize=(12, 8))
plt.imshow(both_lines_rgb)
plt.title(f'LINE DETECTION AFTER TEXT REMOVAL\nHorizontal (Blue): {len(horizontal_lines)}, Vertical (Red): {len(vertical_lines)}', 
          fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()