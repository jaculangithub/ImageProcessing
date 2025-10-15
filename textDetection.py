from PIL import Image, ImageDraw
import pytesseract

# Load the image
img = Image.open('images/atmwithdrawalv2.png')

# Perform OCR and get detailed data including bounding boxes - USE image_to_data() instead of image_to_string()
data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)

# Create a draw object to draw bounding boxes
draw = ImageDraw.Draw(img)

# Iterate through each detected text and its bounding box
for i in range(len(data['text'])):
    # Skip empty text
    if not data['text'][i].strip():
        continue
        
    # Get the bounding box coordinates
    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
    
    # Draw the bounding box
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    
    # Optionally, print the detected text and its coordinates
    print(f"Text: '{data['text'][i]}', Bounding Box: (x={x}, y={y}, w={w}, h={h})")

# Save or display the image with bounding boxes
img.show()  # or img.save('output.png')