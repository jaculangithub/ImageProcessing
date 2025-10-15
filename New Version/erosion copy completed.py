import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Step 0: Load image ----------------
def load_image(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# ---------------- Step 1: Threshold ----------------
def threshold_image(gray, thresh_val=128):
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return thresh

# ---------------- Step 2: Morphology ----------------
def extract_lines(thresh, vert_len=30, horiz_len=30):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    return vertical_lines, horizontal_lines

# ---------------- Step 3: Contour detection ----------------
def get_segments(lines):
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [cv2.boundingRect(c) for c in contours]  # (x, y, w, h)
    return segments

# ---------------- Step 4: Find aligned pairs ----------------
def find_vertical_pairs(segments, y_tol=5, h_tol=5):
    pairs = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(y1 - y2) <= y_tol and abs(h1 - h2) <= h_tol:
                pairs.append((segments[i], segments[j]))
    return pairs

def find_horizontal_pairs(segments, x_tol=5, w_tol=5):
    pairs = []
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(x1 - x2) <= x_tol and abs(w1 - w2) <= w_tol:
                pairs.append((segments[i], segments[j]))
    return pairs

# ---------------- Step 5: Detect rectangles ----------------
def detect_rectangles(vertical_pairs, horizontal_pairs):
    rectangles = []
    for verticalpair in vertical_pairs:
        v1, v2 = verticalpair
        v_left_x = min(v1[0], v2[0])
        v_right_x = max(v1[0], v2[0]) + max(v1[2], v2[2])
        v_top_y = min(v1[1], v2[1])
        v_bottom_y = max(v1[1]+v1[3], v2[1]+v2[3])
        
        for horizontalpair in horizontal_pairs:
            h1, h2 = horizontalpair
            h_top_y = min(h1[1], h2[1])
            h_bottom_y = max(h1[1]+h1[3], h2[1]+h2[3])
            h_left_x = min(h1[0], h2[0])
            h_right_x = max(h1[0]+h1[2], h2[0]+h2[2])
            
            if h_left_x >= v_left_x and h_right_x <= v_right_x and v_top_y >= h_top_y and v_bottom_y <= h_bottom_y:
                h1Width = max(h1[2], h2[2])
                maxWidth = v_right_x - v_left_x
                v1Height = max(v1[3], v2[3])
                maxHeight = h_bottom_y - h_top_y
                
                if((maxWidth - h1Width) < 40 and (maxHeight - v1Height) < 40):
                    rectangles.append((v_left_x, h_top_y, v_right_x, h_bottom_y))
    return rectangles

# ---------------- Step 6: Draw rectangles & remove inner content ----------------
def draw_rectangles(image, rectangles, bg_color=(255, 255, 255)):
    image_rect = image.copy()
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        # Fill inside the rectangle with background color
        image_rect[y1-1:y2+1, x1-1:x2+1] = bg_color
        # Draw bounding rectangle
        # cv2.rectangle(image_rect, (x1-1, y1-1), (x2+1, y2+1), (0, 0, 255), 1)
        # Put index text above rectangle
        # cv2.putText(image_rect, str(i+1), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return image_rect

# ---------------- Step 7: Visualization ----------------
def visualize(vertical_lines, horizontal_lines, image_rect):
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,3,1)
    plt.title("Vertical Lines")
    plt.imshow(vertical_lines, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.title("Horizontal Lines")
    plt.imshow(horizontal_lines, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.title("Rectangles detected in Image")
    plt.imshow(cv2.cvtColor(image_rect, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# ---------------- Step 8: Arrow detection ----------------
def detect_arrows(image_clean, min_area=50, arrow_color=(0,0,255)):
    """
    Detect arrows in the image.
    image_clean: image with rectangles removed/cleaned
    min_area: minimum area to consider a contour as an arrow
    arrow_color: color to draw arrows
    """
    gray = cv2.cvtColor(image_clean, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    arrow_image = image_clean.copy()
    arrows = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h if h != 0 else 0
        # Heuristic: arrows tend to be elongated
        if 0.3 < aspect_ratio < 3:
            arrows.append((x, y, x + w, y + h))
            cv2.rectangle(arrow_image, (x, y), (x+w, y+h), arrow_color, 1)
    
    print("Detected arrows (x1, y1, x2, y2):")
    for arrow in arrows:
        print(arrow)
    
    return arrow_image, arrows

# ---------------- Last 8: Main workflow ----------------
def main(image_path):
    image, gray = load_image(image_path)
    thresh = threshold_image(gray)
    vertical_lines, horizontal_lines = extract_lines(thresh)
    
    vertical_segments = get_segments(vertical_lines)
    horizontal_segments = get_segments(horizontal_lines)
    
    vertical_pairs = find_vertical_pairs(vertical_segments)
    horizontal_pairs = find_horizontal_pairs(horizontal_segments)
    
    rectangles = detect_rectangles(vertical_pairs, horizontal_pairs)
    
    # print("Detected rectangles (x1, y1, x2, y2):")
    # for rect in rectangles:
    #     print(rect)
    
    image_rect = draw_rectangles(image, rectangles)
    
    # Step 9: Detect arrows in cleaned diagram
    arrow_image, arrows = detect_arrows(image_rect)
    
    visualize(vertical_lines, horizontal_lines, image_rect)
    # visualize(vertical_lines, horizontal_lines, arrow_image)

# ---------------- Run ----------------
main("./images/login.png")
