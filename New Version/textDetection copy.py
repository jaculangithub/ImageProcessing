import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import Output

# --- Contour Detection Functions ---
def threshold_image(gray, thresh_val=128):
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV) 
    return thresh

# ---------------- Step 2: Morphology ----------------
def extract_lines(thresh, vert_len=15, horiz_len=15):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    return vertical_lines, horizontal_lines

# ---------------- Step 3: Contour detection ----------------
def get_segments(lines):
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [cv2.boundingRect(c) for c in contours]  # (x, y, w, h)
    return segments

# ---------------- Step 4: Find aligned pairs ----------------
def find_vertical_pairs(segments, y_tol=5, h_tol=5):
    pairs = []
    
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(y1 - y2) <= y_tol and abs(h1 - h2) <= h_tol:
                pairs.append((segments[i], segments[j]))
   
    return pairs

def find_horizontal_pairs(segments, x_tol=5, w_tol=5):
    pairs = []
    
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(x1 - x2) <= x_tol and abs(w1 - w2) <= w_tol:
                pairs.append((segments[i], segments[j]))
    
    return pairs

# ---------------- Step 5: Detect rectangles ----------------
def detect_rectangles(vertical_pairs, horizontal_pairs):
    rectangles = []
    
    for verticalpair in vertical_pairs:
        v1, v2 = verticalpair
        v_left_x = min(v1[0], v2[0])
        v_right_x = max(v1[0], v2[0]) + max(v1[2], v2[2])
        v_top_y = min(v1[1], v2[1])
        v_bottom_y = max(v1[1]+v1[3], v2[1]+v2[3])
        
        for horizontalpair in horizontal_pairs:
            h1, h2 = horizontalpair
            h_top_y = min(h1[1], h2[1])
            h_bottom_y = max(h1[1]+h1[3], h2[1]+h2[3])
            h_left_x = min(h1[0], h2[0])
            h_right_x = max(h1[0]+h1[2], h2[0]+h2[2])
            
            if h_left_x >= v_left_x and h_right_x <= v_right_x and v_top_y >= h_top_y and v_bottom_y <= h_bottom_y:
                h1Width = max(h1[2], h2[2])
                maxWidth = v_right_x - v_left_x
                v1Height = max(v1[3], v2[3])
                maxHeight = h_bottom_y - h_top_y
                
                if((maxWidth - h1Width) < 40 and (maxHeight - v1Height) < 40):
                    rectangles.append((v_left_x, h_top_y, v_right_x, h_bottom_y))
                    
    return rectangles

def remove_detected_contours(original_img):
    """Detect and remove lines/rectangles from the image"""
    # Create a copy of the original image
    result = original_img.copy()
    
    # Apply threshold
    thresh = threshold_image(original_img, 128)
    
    # Extract lines
    vertical_lines, horizontal_lines = extract_lines(thresh)
    
    # Combine all lines
    all_lines = cv2.bitwise_or(vertical_lines, horizontal_lines)
    
    # Remove the detected lines from original image by filling with white
    result[all_lines == 255] = 255
    
    return result

# --- 1. Read grayscale image ---
img = cv2.imread("./images/cd7.png", cv2.IMREAD_GRAYSCALE)

# --- Apply contour detection and removal ---
img_no_contours = remove_detected_contours(img)

# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                             cv2.THRESH_BINARY, 11, 2)
_, img_processed = cv2.threshold(img_no_contours, 128, 255, cv2.THRESH_BINARY_INV)

# --- 2. Noise removal ---
def noise_removal(image):
    # kernel = np.ones((1, 1), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # image = cv2.medianBlur(image, 3)
    return image

clean_img = noise_removal(img_processed)

# --- 3. OCR + bounding box info ---
data = pytesseract.image_to_data(clean_img, output_type=Output.DICT)
d1 = pytesseract.image_to_string(clean_img)

# --- 4. Convert grayscale to BGR for drawing ---
img_boxes = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2BGR)

# --- 5. Draw bounding boxes ---
n_boxes = len(data['level'])
for i in range(n_boxes):
    text = data['text'][i].strip()
    conf = int(data['conf'][i])

    if text != "":  # filter low confidence or empty
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_boxes, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
print(d1)

# --- 6. Show final result ---
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
plt.title("Text Detection with Bounding Boxes (After Contour Removal)")
plt.axis("off")
plt.show()

# --- Show comparison ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img_no_contours, cmap='gray')
plt.title("After Contour Removal")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
plt.title("Final OCR Result")
plt.axis("off")

plt.tight_layout()
plt.show()