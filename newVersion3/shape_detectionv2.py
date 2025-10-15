import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
color_img = cv2.imread('images/atmwithdrawalv2.png')
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# Step 1: Thresholding (Inverted to detect shapes)
_, binary_img = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY_INV)

# Step 2: Find contours with hierarchy information
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Convert hierarchy to more readable format
if hierarchy is None:
    hierarchy = np.array([])
else:
    hierarchy = hierarchy[0]

# Step 3: Filter contours
filtered_contours = []  # Parent contours
child_contours = []     # Valid child contours (not too similar to parent)
contour_areas = [cv2.contourArea(c) for c in contours]

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    
    if i >= len(hierarchy):
        continue
        
    next_idx, prev_idx, first_child_idx, parent_idx = hierarchy[i]
    
    if parent_idx != -1:  # This is a child contour
        parent_area = contour_areas[parent_idx]
        area_ratio = area / parent_area if parent_area > 0 else 0
        
        # Only keep child contours that are NOT too similar to parent
        if area_ratio <= 0.80:
            child_contours.append(c)
        continue
    
    # Filter criteria for parent contours
    if area > 500:
        if aspect_ratio > 0.3 and aspect_ratio < 3.5:
            if max(w, h) / min(w, h) < 4:
                filtered_contours.append(c)

print(f"Number of contours found: {len(contours)}")
print(f"Number of parent contours after filtering: {len(filtered_contours)}")
print(f"Number of child contours after filtering: {len(child_contours)}")

# Print remaining contours
print("\nREMAINING PARENT CONTOURS:")
print("=" * 40)
for i, c in enumerate(filtered_contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    print(f"Parent {i}: Area = {area:.0f} px, Size = {w}x{h} px")

print("\nREMAINING CHILD CONTOURS:")
print("=" * 40)
for i, c in enumerate(child_contours):
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    print(f"Child {i}: Area = {area:.0f} px, Size = {w}x{h} px")

# Create the three required plots
plt.figure(figsize=(18, 6))

# Plot 1: Image Processing (Original + Binary)
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
plt.imshow(binary_img, cmap='gray', alpha=0.3)  # Overlay binary image
plt.title('Image Processing\nOriginal + Binary Overlay', fontsize=14, fontweight='bold')
plt.axis('off')

# Plot 2: All Detected Contours
all_contours_img = color_img.copy()
cv2.drawContours(all_contours_img, contours, -1, (0, 0, 255), 2)  # Red contours
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(all_contours_img, cv2.COLOR_BGR2RGB))
plt.title(f'All Detected Contours\n({len(contours)} contours)', fontsize=14, fontweight='bold')
plt.axis('off')

# Plot 3: Remaining Parent (Green) and Child (Red) Contours
final_img = color_img.copy()

# Draw parent contours in GREEN
for i, c in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawContours(final_img, [c], -1, (0, 255, 0), 2)
    
    area = cv2.contourArea(c)
    cv2.putText(final_img, f"P:{int(area)}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw child contours in RED
for i, c in enumerate(child_contours):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(final_img, [c], -1, (0, 0, 255), 2)
    
    area = cv2.contourArea(c)
    cv2.putText(final_img, f"C:{int(area)}", (x, y+h+15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title(f'Filtered Contours\nParents (Green): {len(filtered_contours)}, Children (Red): {len(child_contours)}', 
          fontsize=14, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.savefig('contour_analysis_simple.png', dpi=300, bbox_inches='tight')
plt.show()

# Display final result with larger view
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.title(f'FINAL RESULT: Parent Contours (Green) and Child Contours (Red)\nTotal: {len(filtered_contours) + len(child_contours)} contours remaining', 
          fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()