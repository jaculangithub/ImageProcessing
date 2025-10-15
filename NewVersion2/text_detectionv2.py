import pytesseract
from PIL import Image
import cv2
import numpy as np

img_file = "./images/sample.png"
no_noise = "temp/no_noise.png"

# Load image with PIL
img_pil = Image.open(img_file)

# Convert PIL Image to OpenCV format (numpy array)
img_cv = np.array(img_pil)
# Convert RGB to BGR (OpenCV uses BGR format)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

# Perform OCR on original image
ocr_result = pytesseract.image_to_string(img_pil)
print("Actual Image: " + ocr_result)

# Now invert the image using OpenCV
inverted_image = cv2.bitwise_not(img_cv)
cv2.imwrite("NewVersion2/inverted.png", inverted_image)

# Binarization 
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_img = grayscale(img_cv)
cv2.imwrite("NewVersion2/gray_img.png", gray_img)
