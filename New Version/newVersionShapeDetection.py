import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load Image ---
img = cv2.imread("./images/atmwithdrawalv2.png")   # <-- replace with your UML image path
img_orig = img.copy()

# --- Step 1: Grayscale ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Step 2: Threshold (invert: black lines → white) ---
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# --- Step 3: Edge Detection ---
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# --- Step 4: Line Detection ---
line_img = img.copy()
detected_lines = []
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=5)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0,0,255), 2)
        detected_lines.append(((x1, y1), (x2, y2)))

# --- Step 5: Shape Detection (Rectangles, Ellipses, etc.) ---
shape_img = img.copy()
detected_shapes = []
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    if len(approx) == 4:  # Rectangle (Class/Interface box)
        cv2.rectangle(shape_img, (x,y), (x+w, y+h), (0,255,0), 2)
        detected_shapes.append({"type": "rectangle", "bbox": (x, y, w, h)})
    elif len(approx) > 8:  # Ellipse (Use Case, Start/End)
        cv2.ellipse(shape_img, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (255,0,0), 2)
        detected_shapes.append({"type": "ellipse", "bbox": (x, y, w, h)})
    elif len(approx) == 6:  # Possible diamond (Decision)
        cv2.drawContours(shape_img, [approx], 0, (255,255,0), 2)
        detected_shapes.append({"type": "diamond", "bbox": (x, y, w, h)})

# --- Step 6: Combined Detection ---
combined_img = img.copy()
if lines is not None:
    for (x1, y1), (x2, y2) in detected_lines:
        cv2.line(combined_img, (x1, y1), (x2, y2), (0,0,255), 2)
for shape in detected_shapes:
    x, y, w, h = shape["bbox"]
    if shape["type"] == "rectangle":
        cv2.rectangle(combined_img, (x,y), (x+w, y+h), (0,255,0), 2)
    elif shape["type"] == "ellipse":
        cv2.ellipse(combined_img, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (255,0,0), 2)
    elif shape["type"] == "diamond":
        cv2.rectangle(combined_img, (x,y), (x+w, y+h), (255,255,0), 2)

# --- Visualization of Steps ---
titles = ["Original", "Grayscale", "Thresholded", "Edges", "Lines Detected", "Shapes Detected", "Combined"]
images = [img_orig, gray, thresh, edges, line_img, shape_img, combined_img]

plt.figure(figsize=(15,10))
for i in range(len(images)):
    plt.subplot(2,4,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# --- Print Detected Shapes and Lines ---
print("\n--- Detected Shapes ---")
for s in detected_shapes:
    print(f"{s['type']} at {s['bbox']}")

print("\n--- Detected Lines ---")
for l in detected_lines:
    print(f"Line from {l[0]} to {l[1]}")
