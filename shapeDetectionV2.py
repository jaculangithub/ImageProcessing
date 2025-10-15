import cv2
import numpy as np
import matplotlib.pyplot as plt

Gray_image = cv2.imread('images/atmwithdrawal.png', 0)
# Gray_image = cv2.imread('images/atmwithdrawalv2.png', 0)

# display
plt.figure(figsize=(10, 10))
plt.imshow(Gray_image, cmap='gray'); plt.title('Original Image'); plt.axis('on')
plt.show() 