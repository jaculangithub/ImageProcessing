from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

# Load the image
img = Image.open('./images/sample.png')
img_rgb = img.convert('RGB')  # Ensure RGB mode for drawing

# Perform OCR and get detailed data including bounding boxes
data = pytesseract.image_to_data(img_rgb, lang="eng", output_type=pytesseract.Output.DICT)

# Create a copy for drawing bounding boxes
img_with_boxes = img_rgb.copy()
draw_boxes = ImageDraw.Draw(img_with_boxes)

# Create a mask image where detected text will be white
mask_img = Image.new('RGB', img_rgb.size, (0, 0, 0))  # Black background
draw_mask = ImageDraw.Draw(mask_img)

# Lists to store detected text information
detected_texts = []

# Iterate through each detected text and its bounding box
for i in range(len(data['text'])):
    # Skip empty text and low confidence detections
    # if not data['text'][i].strip() or int(data['conf'][i]) < 30:
    #     continue
        
    # Get the bounding box coordinates
    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
    
    # Draw the bounding box on original image (red)
    draw_boxes.rectangle([x, y, x + w, y + h], outline="red", width=2)
    
    # Draw white rectangle and text on mask image
    draw_mask.rectangle([x, y, x + w, y + h], fill="white", outline="white")
    
    # Store detected text information
    detected_texts.append({
        'text': data['text'][i],
        'bbox': (x, y, w, h),
        'confidence': data['conf'][i]
    })
    
    # Print the detected text and its coordinates
    print(f"Text: '{data['text'][i]}', Confidence: {data['conf'][i]}, BBox: (x={x}, y={y}, w={w}, h={h})")

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image with bounding boxes
axes[0].imshow(np.array(img_with_boxes))
axes[0].set_title('Original Image with Bounding Boxes (Red)')
axes[0].axis('off')

# Mask image with white text areas
axes[1].imshow(np.array(mask_img))
axes[1].set_title('Text Areas (White)')
axes[1].axis('off')

# Combined visualization: Original image with white text overlay
combined_img = img_rgb.copy()
draw_combined = ImageDraw.Draw(combined_img)

for text_info in detected_texts:
    x, y, w, h = text_info['bbox']
    # Draw white rectangle over text area
    draw_combined.rectangle([x, y, x + w, y + h], fill="white", outline="white")

axes[2].imshow(np.array(combined_img))
axes[2].set_title('Original with White Text Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Print summary
print(f"\nSummary:")
print(f"Total text elements detected: {len(detected_texts)}")
print(f"Detected texts: {[text_info['text'] for text_info in detected_texts]}")