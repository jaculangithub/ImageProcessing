import cv2
import pytesseract
import matplotlib.pyplot as plt

# Path to tesseract executable (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
image = cv2.imread("./images/sample.png")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive threshold (better than fixed threshold)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
)

# Detect text with bounding boxes
data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)

# Draw bounding boxes (only for confident text)
for i in range(len(data['text'])):
    if data['text'][i].strip() != "" and int(data['conf'][i]) > 20:  # stricter filter
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, data['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 2)

# Plot the result
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Text with Bounding Boxes")
plt.axis("off")
plt.show()

# Print detected text
detected_words = [word for word in data['text'] if word.strip() != ""]
print("Detected Text:")
print(" ".join(detected_words))
