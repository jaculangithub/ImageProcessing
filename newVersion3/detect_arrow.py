import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """
    Loads an image and converts it to a clean binary version.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} not loaded. Check the path.")
    
    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding for varying lighting conditions
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological closing to close small holes inside shapes
    kernel_close = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # Morphological opening to remove small noise points
    kernel_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    
    return img, binary

def separate_connected_components(binary_img, original_img):
    """
    The core function to separate connected components using the Watershed algorithm.
    Returns a list of masks for each separated object.
    """
    # Step 1: Sure background area via dilation
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(binary_img, kernel, iterations=2)
    
    # Step 2: Sure foreground area via distance transform + thresholding
    dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    
    # Normalize the distance transform for easier thresholding
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Adjust this threshold (0.3-0.7) to control what is considered "sure foreground"
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Step 3: Find the unknown region (boundary area)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Step 4: Label markers for the sure foreground
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply Watershed algorithm. This modifies the original image in-place.
    img_for_watershed = original_img.copy()
    markers = cv2.watershed(img_for_watershed, markers)
    
    # After watershed, boundaries are marked with -1, and objects have values >= 2
    # Let's create a mask for each separate object
    separated_masks = []
    for marker_val in np.unique(markers):
        if marker_val > 1:  # Skip background (-1, 1) and unknown (0)
            mask = np.zeros_like(binary_img, dtype=np.uint8)
            mask[markers == marker_val] = 255
            separated_masks.append(mask)
            
    return separated_masks, markers

def is_arrow(contour, mask_area):
    """
    Heuristic function to determine if a contour is likely an arrow.
    This is crucial for UML diagram interpretation.
    """
    area = cv2.contourArea(contour)
    if area < 100:  # Too small to be a significant arrow
        return False
        
    # Get the bounding rectangle and aspect ratio
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # Arrows are often long and thin, or have a triangular head
    # Check for high aspect ratio (long horizontal arrow) or low (long vertical arrow)
    is_long_and_thin = aspect_ratio > 4.0 or aspect_ratio < 0.25
    
    # Check compactness (how close the shape is to a circle)
    # Arrows are less compact than boxes or circles.
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter * perimeter)
    else:
        compactness = 0
        
    is_non_compact = compactness < 0.2  # Circles have compactness ~1, lines ~0
    
    # Check for a triangular tip using contour approximation
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    has_pointy_tip = len(approx) >= 3 and len(approx) <= 7  # Triangles/pentagons for arrowheads
    
    # Check convexity defects - an arrowhead often has significant defects
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            # Check if any defect is deep enough to be an arrowhead
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 10000:  # This value is distance-dependent, may need tuning
                    has_pointy_tip = True
                    break
    
    # A shape is likely an arrow if it's long and thin OR has a pointy tip
    return (is_long_and_thin or has_pointy_tip) and is_non_compact

def process_uml_diagram(image_path):
    """
    Main function to process a UML diagram image, separate connected components,
    and identify arrows.
    """
    # 1. Preprocess the image
    original_img, binary_img = preprocess_image(image_path)
    result_img = original_img.copy()
    
    # 2. Find initial contours to see connected objects
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Found {len(contours)} initial contour(s)")
    
    # 3. We'll store final results here
    arrow_contours = []
    shape_contours = []
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore very small artifacts
            continue
            
        # Create a mask for this specific contour
        contour_mask = np.zeros_like(binary_img)
        cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
        
        # Check if this might be multiple connected objects
        # Try to separate using watershed on this specific contour region
        separated_masks, markers = separate_connected_components(contour_mask, original_img)
        
        print(f"Contour {i+1} separated into {len(separated_masks)} component(s)")
        
        # Analyze each separated component
        for mask in separated_masks:
            # Find the contour of this separated component
            comp_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if comp_contours:
                comp_cnt = comp_contours[0]
                
                # Classify as arrow or shape
                if is_arrow(comp_cnt, cv2.countNonZero(mask)):
                    arrow_contours.append(comp_cnt)
                    # Draw arrow in green
                    cv2.drawContours(result_img, [comp_cnt], -1, (0, 255, 0), 2)
                else:
                    shape_contours.append(comp_cnt)
                    # Draw shape in red
                    cv2.drawContours(result_img, [comp_cnt], -1, (0, 0, 255), 2)
    
    # 4. Display and save results
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('Original UML Diagram')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(binary_img, cmap='gray')
    plt.title('Binary Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Arrows (Green): {len(arrow_contours)}\nShapes (Red): {len(shape_contours)}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('uml_analysis_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return arrow_contours, shape_contours, result_img

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    image_path = "./images/login.png"  # <-- REPLACE WITH YOUR IMAGE PATH
    
    try:
        arrows, shapes, result_image = process_uml_diagram(image_path)
        print(f"Analysis complete! Found {len(arrows)} arrows and {len(shapes)} shapes.")
        
        # Save the result image
        cv2.imwrite("detected_arrows_and_shapes.jpg", result_image)
        
    except Exception as e:
        print(f"Error processing image: {e}")