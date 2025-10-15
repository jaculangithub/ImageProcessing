import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

def detect_rectangles(img_path, output_dir="NewVersion2"):
    # Step 1: Read input image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found. Check the file path.")
    
    # Make sure output folder exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Save as BMP (optional)
    bmp_path = os.path.join(output_dir, "output.bmp")
    cv2.imwrite(bmp_path, img)
    
    # Create a copy for processing
    processed = img.copy()
    
    # Step 3: Convert to grayscale
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Preprocessing - Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 5: Adaptive thresholding for better handling of lighting variations
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Step 6: Morphological operations to close gaps and smooth edges
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Step 7: Detect contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Copy image for drawing
    detected = img.copy()
    
    # List to store detected rectangles
    rectangles = []
    
    # Filter parameters
    min_area = 500  # Minimum area to consider
    max_area = img.shape[0] * img.shape[1] * 0.8  # Maximum area to consider (80% of image)
    aspect_ratio_range = (0.2, 5.0)  # Valid aspect ratio range
    
    for cnt in contours:
        # Calculate contour area and filter by size
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
            
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4:
            # Check convexity
            if not cv2.isContourConvex(approx):
                continue
                
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Filter by aspect ratio
            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                continue
                
            # Check if the quadrilateral is reasonably rectangular
            # by measuring the angles between adjacent sides
            vectors = []
            for i in range(4):
                pt1 = approx[i][0]
                pt2 = approx[(i+1) % 4][0]
                vectors.append((pt2[0] - pt1[0], pt2[1] - pt1[1]))
            
            # Calculate angles between adjacent vectors
            angles = []
            for i in range(4):
                v1 = vectors[i]
                v2 = vectors[(i+1) % 4]
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                if mag1 * mag2 > 0:
                    cos_angle = dot / (mag1 * mag2)
                    cos_angle = np.clip(cos_angle, -1, 1)  # Ensure valid range
                    angle = np.degrees(np.arccos(cos_angle))
                    angles.append(angle)
            
            # Check if angles are close to 90 degrees
            if all(80 <= angle <= 100 for angle in angles):
                rectangles.append(approx)
                cv2.drawContours(detected, [approx], -1, (0, 255, 0), 3)
    
    # Step 8: Show results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(thresh, cmap="gray")
    axes[0, 1].set_title("Threshold Image")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(morphed, cmap="gray")
    axes[1, 0].set_title("Morphological Operations")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"Detected Rectangles: {len(rectangles)}")
    axes[1, 1].axis("off")
    
    print(f"Number of rectangles detected: {len(rectangles)}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "detection_process.png"))
    plt.show()
    
    # Save the result image
    cv2.imwrite(os.path.join(output_dir, "detected_rectangles.jpg"), detected)
    
    return detected, rectangles

# Run the detection
if __name__ == "__main__":
    img_path = "./images/shape.jpg"  # change to your file name
    detected_img, rectangles = detect_rectangles(img_path)