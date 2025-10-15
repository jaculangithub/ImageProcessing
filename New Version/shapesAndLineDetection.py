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

# --- Visualization of Steps ---
titles = ["Original", "Grayscale", "Thresholded", "Edges", "Lines Detected"]
images = [img_orig, gray, thresh, edges, line_img]

plt.figure(figsize=(15,8))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# --- Print Detected Lines ---
print("\n--- Detected Lines ---")
for l in detected_lines:
    print(f"Line from {l[0]} to {l[1]}")
