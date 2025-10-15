import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from pytesseract import Output

# --- 1. Read grayscale image ---
img = cv2.imread("./images/atm.png", cv2.IMREAD_GRAYSCALE)

# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                             cv2.THRESH_BINARY, 11, 2)
_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
# --- 2. Noise removal ---
def noise_removal(image):
    # kernel = np.ones((1, 1), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    # image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # image = cv2.medianBlur(image, 3)
    return image

clean_img = noise_removal(img)

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
plt.title("Text Detection with Bounding Boxes")
plt.axis("off")
plt.show()
