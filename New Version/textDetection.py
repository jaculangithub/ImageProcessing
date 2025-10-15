import cv2
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
import numpy as np

# ----------- CONFIGURE TESSERACT PATH (Windows only) -----------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------- IMAGE PREPROCESSING FUNCTION -----------
def preprocess_for_ocr(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise removal
    denoised = cv2.medianBlur(gray, 3)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Thresholding
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return processed

# ----------- STEP 1: LOAD AND PREPROCESS IMAGE -----------
image_path = "./images/cd7.png"  # change this to your image file
img = cv2.imread(image_path)

# Preprocess the image
processed_img = preprocess_for_ocr(img)

# ----------- STEP 2: EXTRACT TEXT DATA WITH BOUNDING BOXES -----------
# Use processed image for OCR
data = pytesseract.image_to_data(processed_img, output_type=Output.DICT)

# Convert processed image back to RGB for visualization
img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)

# Draw bounding boxes and text
for i in range(len(data['text'])):
    text = data['text'][i]
    conf = int(data['conf'][i])
    if conf > 30 and text.strip() != "":  # confidence filter
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_rgb, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# ----------- STEP 3: EXTRACT TEXT -----------
extracted_text = pytesseract.image_to_string(processed_img)

# ----------- STEP 4: VISUALIZE IMAGE WITH BOUNDING BOXES -----------
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.title("Detected Text with Bounding Boxes\nExtracted Text Displayed Above Boxes", fontsize=14)
plt.axis("off")

# Add extracted text as annotation
plt.figtext(0.5, 0.01, f"Extracted Text:\n{extracted_text}", 
           ha="center", fontsize=10, 
           bbox={"facecolor":"orange", "alpha":0.2, "pad":5},
           wrap=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Make space for text
plt.show()

# ----------- PRINT TEXT IN CONSOLE -----------
print("\n--- Extracted Text ---\n")
print(extracted_text)