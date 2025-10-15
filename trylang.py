import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pytesseract

# Load the same image for both processes
image_path = 'images/shape.jpg'

# OCR TEXT DETECTION
img_pil = Image.open(image_path)
draw = ImageDraw.Draw(img_pil)
data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DICT)

# Store text bounding boxes
text_boxes = []
for i in range(len(data['text'])):
    if not data['text'][i].strip():
        continue
    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    text_boxes.append((x, y, x+w, y+h))  # Store as (x1,y1,x2,y2)
    print(f"Text: '{data['text'][i]}', Box: (x={x}, y={y}, w={w}, h={h})")

ocr_result = np.array(img_pil)

# CONTOUR DETECTION
color_img = cv2.imread(image_path)
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create original contours image (excluding first contour)
original_contours_img = color_img.copy()
if len(contours) > 0:
    cv2.drawContours(original_contours_img, contours[1:], -1, (0, 255, 0), 2)
original_contours_result = cv2.cvtColor(original_contours_img, cv2.COLOR_BGR2RGB)

# Filter out contours inside text boxes
filtered_contours = []
for c in contours[1:] if len(contours) > 0 else []:
    x, y, w, h = cv2.boundingRect(c)
    center_x, center_y = x + w/2, y + h/2  # Get contour center point
    
    # Check if center is inside any text box
    inside_text = False
    for (tx1, ty1, tx2, ty2) in text_boxes:
        if tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2:
            inside_text = True
            break
    
    if not inside_text:
        filtered_contours.append(c)

print(f"Original contours: {len(contours)-1 if len(contours) > 0 else 0}, After removal: {len(filtered_contours)}")

# Draw filtered contours
filtered_contours_img = color_img.copy()
for c in filtered_contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(filtered_contours_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
filtered_contours_result = cv2.cvtColor(filtered_contours_img, cv2.COLOR_BGR2RGB)

# DISPLAY RESULTS
plt.figure(figsize=(20, 7))

# Subplot 1: OCR Text Detection
plt.subplot(1, 3, 1)
plt.imshow(ocr_result)
plt.title('OCR Text Detection\n(Red Boxes)')
plt.axis('off')

# Subplot 2: Original Contours (excluding first)
plt.subplot(1, 3, 2)
plt.imshow(original_contours_result)
plt.title(f'Original Contours\n(Green, count: {len(contours)-1 if len(contours) > 0 else 0})')
plt.axis('off')

# Subplot 3: Filtered Contours
plt.subplot(1, 3, 3)
plt.imshow(filtered_contours_result)
plt.title(f'Filtered Contours\n(Green, count: {len(filtered_contours)})')
plt.axis('off')

plt.tight_layout()
plt.show()