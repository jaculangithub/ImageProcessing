import cv2
import matplotlib.pyplot as plt

# --- Load image ---
img = cv2.imread("./images/atmwithdrawal.png")   # replace with your image path

# --- Convert to Grayscale ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Convert to Binary (Black & White) using Thresholding ---
# Any pixel > 127 becomes white (255), else black (0)
_, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# --- Visualization ---
titles = ["Original (Color)", "Grayscale", "Black & White (Thresholded)"]
images = [img, gray, bw]

plt.figure(figsize=(12,4))
for i in range(3):
    plt.subplot(1,3,i+1)
    if len(images[i].shape) == 2:  # grayscale or bw
        plt.imshow(images[i], cmap="gray")
    else:  # color image
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
