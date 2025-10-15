import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 
import pytesseract
from pytesseract import Output

epsilon_ratio = 0.04

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
def extract_lines(thresh, vert_len=15, horiz_len=15):
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
def detect_rectangles(vertical_pairs, horizontal_pairs, vertical_lines, horizontal_lines):
    rectangles = []
    line_areas_to_remove = []  # Store the actual line areas to remove
    
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
                    
                    # ADDED: Store the actual line areas that form this rectangle
                    # Top horizontal line area
                    line_areas_to_remove.append((h_left_x, h_top_y, h_right_x, h_top_y + max(h1[3], h2[3])))
                    # Bottom horizontal line area  
                    line_areas_to_remove.append((h_left_x, h_bottom_y - max(h1[3], h2[3]), h_right_x, h_bottom_y))
                    # Left vertical line area
                    line_areas_to_remove.append((v_left_x, v_top_y, v_left_x + max(v1[2], v2[2]), v_bottom_y))
                    # Right vertical line area
                    line_areas_to_remove.append((v_right_x - max(v1[2], v2[2]), v_top_y, v_right_x, v_bottom_y))
                    
    return rectangles, line_areas_to_remove

# ---------------- Modified: Draw rectangles and remove lines that probably a circle ----------------
def remove_rectangles(image, line_areas_to_remove, vLines, hLines, bg_color=(255, 255, 255)):
    image_rect = image.copy()
    
    # Remove the actual line areas that form the rectangles
    for line_area in line_areas_to_remove:
        x1, y1, x2, y2 = line_area
        # Fill the actual line area with background color
        image_rect[y1:y2, x1:x2] = bg_color

    for vLine in vLines:
        x, y, w, h = vLine
        
        # For vertical lines: width should be much smaller than height
        aspect_ratio = w / h
        if aspect_ratio > 0.7:  # This is a proper vertical line
            middle_y = y + h // 2
            
            image_rect[middle_y-3:middle_y+3, x-3] = bg_color       #remove left 
            image_rect[middle_y-3:middle_y+3,x+w+3] = bg_color      #remove right
        
        
              
    for hLine in hLines:
        x, y, w, h = hLine
        
        # For horizontal lines: height should be much smaller than width
        aspect_ratio = h / w
        if aspect_ratio > 0.7:  # This is a proper horizontal line
         
            middle_x = x + w // 2
                    
            image_rect[y-3, middle_x-3:middle_x+3] = bg_color   #remove top
            image_rect[y+h+3, middle_x-3:middle_x+3] = bg_color #remove bottom
    
    return image_rect


# new code 9/27/2025 12AM
# def classify_relationship(contours, hierarchy, idx):
#     """
#     Classify a contour (line + attached symbol) into UML relationship type.
#     Returns classification and points (start/end for directed, endpoints for plain associations).
#     """
    
#     global epsilon_ratio
#     cnt = contours[idx]
#     child_idx = hierarchy[0][idx][2]
#     has_five_seq, convexHull = None, None
#     epsilon_ratio = 0.04
#     has_five_seq, convexHull = None, None
#     while True:
#         # Get approximate points for the contour
#         epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#         vertices = len(approx)
        
#         # Find two farthest points (endpoints for all line types)
#         farthest_points = find_farthest_points(approx)
#         if farthest_points is None:
#             return "Association (plain line)", None, None, None, None, approx
        
#         point1, point2, max_distance = farthest_points
        
#             # If no 5-point sequence, check if it has a child symbol
#         if child_idx != -1:
#             child_cnt = contours[child_idx]
#             child_epsilon = 0.08 * cv2.arcLength(child_cnt, True)
#             child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
#             child_vertices = len(child_approx)
#             # print("Points: ", child_approx.reshape(-1, 2))
#             # Classify symbol type
#             if child_vertices == 3:
#                 relationship_type = "inheritance"
#             elif child_vertices == 4:
#                 relationship_type = "aggregation"
#             else:
#                 relationship_type = "Other unfilled shape"
            
#             # Determine start/end: which farthest point is closer to child symbol
#             start_point, end_point = determine_start_end_with_child(point1, point2, child_approx)
            
#             return relationship_type, start_point, end_point, point1, point2, approx
        
              
#         # First, check for 5-point sequence (this takes priority)
#         has_five_seq, convexHull = find_five_point_sequence(approx, point1, point2)
        
#         if has_five_seq: #break if have 5-point sequence,
#             # print("meron")
#             break
#         # if the decimal is exceeding to the bsta pag lumagpas
#         elif epsilon_ratio <= 0.005:
#             # print("greatdr", len(approx))
#             break 
        
#         epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx
    
#     # print("Classifying, Current Epsilon: ", epsilon_ratio)
    
#     if has_five_seq:
#         if convexHull and convexHull == 3:
#             relationship_type = "directed association"
#         else:
#             relationship_type = "composition"
#         # Determine start/end for 5-point sequence relationships
#         start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
#         # print("Start", start_point, " End: ", end_point, " Point1: ", point1, " Point2: ", point2)
#         return relationship_type, start_point, end_point, point1, point2, approx
#     else:
#         # If no 5-point sequence and no child symbol, it's a plain association
#         relationship_type = "association"
#         # Return endpoints but no start/end direction
#         return relationship_type, None, None, point1, point2, approx

def find_farthest_points(approx_points):
    """Find the two points with maximum distance in a set of points."""
    points = np.squeeze(approx_points)  # Handle (N,1,2) → (N,2)
    
    if len(points) < 2:
        return None
    
    max_distance = 0
    point1 = None
    point2 = None
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > max_distance:
                max_distance = distance
                point1 = tuple(points[i])
                point2 = tuple(points[j])
    
    return point1, point2, max_distance

# def determine_start_end_with_child(point1, point2, child_approx):
#     """Determine start/end for contours with child symbols."""
#     child_points = np.squeeze(child_approx)
    
#     if len(child_points) == 0:
#         return point1, point2  # Fallback
    
#     # Calculate distances from each point to the child symbol
#     dist1 = min([np.linalg.norm(np.array(point1) - child_point) for child_point in child_points])
#     dist2 = min([np.linalg.norm(np.array(point2) - child_point) for child_point in child_points])
    
#     # The point closer to the child symbol is the end (arrowhead/diamond side)
#     if dist1 < dist2:
#         return point2, point1  # point1 is end (closer to symbol)
#     else:
#         return point1, point2  # point2 is end (closer to symbol)

def determine_start_end_five_sequence(point1, point2, approx):
    """Determine start/end for 5-point sequence (filled diamond)."""
    points = np.squeeze(approx)
    
    # Find the 5-point sequence
    five_seq_points, _ = find_five_point_sequence(points, point1, point2)
    
    if five_seq_points is None:
        return point1, point2  # Fallback
    
    # Check which of the two farthest points is in the 5-point sequence
    point1_in_seq = any(np.array_equal(np.array(point1), seq_point) for seq_point in five_seq_points)
    point2_in_seq = any(np.array_equal(np.array(point2), seq_point) for seq_point in five_seq_points)
    
    if point1_in_seq and point2_in_seq:
        print("Both meron")
    
    # The point in the sequence is the end (diamond side)
    if point1_in_seq and not point2_in_seq:
        return point2, point1  # point1 is end
    elif point2_in_seq and not point1_in_seq:
        return point1, point2  # point2 is end
    else:
        # Both or neither in sequence, return as plain association
        print("Both wala")
        return None, None

def find_five_point_sequence(points, point1, point2, min_tolerance=5, tolerance_percent=0.25):
    """Find and return the 5-point sequence if it exists."""
    n = len(points)
    if n < 5:
        return None, None
    
    for start in range(n):
        distances = []
        seq_points = []
        
        # Get the 5-point sequence
        for i in range(5):
            p1 = points[(start + i) % n]
            seq_points.append(p1)
        
        # ✅ Check if the two farthest points are in this sequence BEFORE distance calculations
        point1_in_seq = any(np.array_equal(np.array(point1), seq_point) for seq_point in seq_points)
        point2_in_seq = any(np.array_equal(np.array(point2), seq_point) for seq_point in seq_points)
        
        # If both farthest points are in this sequence, skip it
        if point1_in_seq and point2_in_seq:
            # print(f"Skipping sequence at start {start} - contains both farthest points")
            continue  # ✅ Skip to next sequence
        
        # Only calculate distances if we didn't skip
        for i in range(4):
            p1 = seq_points[i]
            p2 = seq_points[i + 1]
            dist = np.linalg.norm(p1 - p2)
            distances.append(dist)
        
        max_dist = max(distances)
        min_dist = min(distances)
        avg_distance = np.mean(distances)
         
        # ratio-scaling 
        original_max = 30
        original_min = 18
        scale_factor = max_dist / original_max
        scaled_min = original_min * scale_factor
        
        dynamic_tolerance = max_dist - scaled_min
        
        if max(distances) - min(distances) <= dynamic_tolerance+5:
            seq_np = np.array(seq_points, dtype=np.int32)
            hull = cv2.convexHull(seq_np, returnPoints=True)
            hull_count = len(hull)
            # print("Found 5 points - L267")
            # print(f"Points: {seq_points}, Distances: {distances}")
            # print(f"Hull Count: {hull_count}, Max: {max(distances):.2f}, Min: {min(distances):.2f}")
            # print("Tolerance1: ", max(distances)-min(distances))
            # print("Tolerance: ",  dynamic_tolerance)
            if hull_count >= 4:
                return seq_points, hull_count
            elif hull_count == 3:
                return seq_points, hull_count
            
    # print("No 5points found that don't contain both farthest points")
    return None, None


def classify_based_on_child(contours, hierarchy, idx):
    """
    Classify contour based on its child contour for activity diagrams
    """
    cnt = contours[idx]
    epsilon = 0.08 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # If no diamond child found, do other activity diagram classifications
    children = []
    child_idx = hierarchy[0][idx][2]
    while child_idx != -1:
        children.append(child_idx)
        child_idx = hierarchy[0][child_idx][0]
    
    grand_children = []
    for child in children:
        grand_child_idx = hierarchy[0][child][2]
        while grand_child_idx != -1:
            grand_children.append(grand_child_idx)
            grand_child_idx = hierarchy[0][grand_child_idx][0]
    
    # Other classifications based on child count
    if len(children) == 4:
        return "destruction node", None, None, approx
    elif len(children) == 1 and len(grand_children) > 0:
        return "end node", None, None, approx
    
    # Check all child contours for diamond shape (4 points)
    child_idx = hierarchy[0][idx][2]
    while child_idx != -1:
        child_cnt = contours[child_idx]
        child_epsilon = 0.08 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
        
        # If child has 4 points (diamond shape)
        if len(child_approx) == 4:
            print("Found diamond child - directed association")
            return "diamond node", None, None, approx
        
        # Move to next sibling child
        child_idx = hierarchy[0][child_idx][0]
    
    
    
    # Default for contours with children but no specific shape
    return "activity node", None, None, approx
    
    
def classify_contour(contours, hierarchy, idx):
    
    global epsilon_ratio
    
    child_idx = hierarchy[0][idx][2]
    
    if child_idx != -1: # has child, likely relationship
        print("has child")
        # return None, None, None, None
        return classify_based_on_child(contours, hierarchy, idx)
    
    else: 

        epsilon_ratio = 0.04
        cnt = contours[idx]
        has_five_seq, convexHull = None, None
        approx = None
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        bbox_area = w * h
        
        # Calculate circularity (for start/end nodes)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        print(" area: ", area, "B area: ", bbox_area, " Circularity ", circularity)
        # Check for circular shape (start node)
        if bbox_area > 0 and (area / bbox_area) >= 0.75 and circularity >= 0.7:
            print("start node")
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)   
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            return "start node", None, None, approx
        
        # likey arrow
        while True:
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)   
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # find two farthest points
            farthest_points = find_farthest_points(approx)
            if farthest_points is None:
                return "Unknown", None, None, approx
            
            point1, point2, max_distance = farthest_points 
            # check if the approx has 5points sequence 
            has_five_seq, convexHull = find_five_point_sequence(approx, point1, point2)

            if has_five_seq: #break if have 5-point sequence,
                # print("meron likely an arrow")
                break
            
            elif epsilon_ratio <= 0.005:
                print("greatdr", len(approx))
                break

            epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx

        if has_five_seq:
            if convexHull and convexHull == 3:
                print("directed arrow")
                classification = "directed arrow"
                start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
                
                return classification, start_point, end_point, approx
            else: 
                print("line")
                classification = "line" 
                return "line", None, None, approx
        
        print("unknown")
        return "line", None, None, approx
        
        
# new code 10/1/2024 1AM - improved merging logic to handle chains and prevent arrow head merging
def detect_contours(image, rectangles, bg_color=0, min_area=100):

    threshImg = threshold_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    detectedContours = threshImg.copy()
    detectText = threshImg.copy()
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    ) 
    # finding a contour with diamond child
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Traverse **all** children instead of just the first one
        child_idx = hierarchy[0][i][2]
        while child_idx != -1:
            child_cnt = contours[child_idx]
            cx, cy, cw, ch = cv2.boundingRect(child_cnt)

            # Filter small noise
            if cw > 25 and ch > 25:
                epsilon = 0.08 * cv2.arcLength(child_cnt, True)
                approx = cv2.approxPolyDP(child_cnt, epsilon, True)

                # Check if it's inside any rectangle and in a corner
                inTheRect = False
                is_in_corner = False
                for rect in rectangles:
                    rx1, ry1, rx2, ry2 = rect
                    if cx >= rx1 and cy >= ry1 and (cx + cw) <= rx2 and (cy + ch) <= ry2:
                        inTheRect = True
                        corner_size = 10
                        if (
                            (cx <= rx1 + corner_size and cy <= ry1 + corner_size) or
                            (cx + cw >= rx2 - corner_size and cy <= ry1 + corner_size) or
                            (cx <= rx1 + corner_size and cy + ch >= ry2 - corner_size) or
                            (cx + cw >= rx2 - corner_size and cy + ch >= ry2 - corner_size)
                        ):
                            is_in_corner = True
                        break

                if inTheRect and is_in_corner:
                    cv2.drawContours(detectText, [cnt], -1, bg_color, -1)

                # Only slice if shape is diamond-like
                if len(approx) == 4:
                    # Remove only middle 5 pixels instead of entire sides
                    # Calculate middle points
                    middle_y = cy + ch // 2
                    middle_x = cx + cw // 2

                    # Remove middle 5 pixels on left and right sides (vertical removal)
                    image[middle_y-3:middle_y+3, cx-4] = (255, 255, 255)  # remove left middle
                    image[middle_y-3:middle_y+3, cx+cw+4] = (255, 255, 255)  # remove right middle

                    # Remove middle 5 pixels on top and bottom sides (horizontal removal)  
                    image[cy-4, middle_x-3:middle_x+3] = (255, 255, 255)  # remove top middle
                    image[cy+ch+4, middle_x-3:middle_x+3] = (255, 255, 255)  # remove bottom middle

            # move to next sibling child
            child_idx = hierarchy[0][child_idx][0]

    
    threshImg = threshold_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    detectedContours = threshImg.copy()
    detectText = threshImg.copy()
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    ) 

    contour_img = cv2.cvtColor(detectedContours, cv2.COLOR_GRAY2BGR)
    contours_info = []
    count = 0
    # find all the contours that are inside the rectangles and remove/skip them
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # check if the contour is inside the rect
        inTheRect = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            # Check if contour bbox is within rectangle bbox
            if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                # Fill rectangle area with background color to remove it
                inTheRect = True
                # check if the contour is in the corner of the bounding box
                # Define corner size (adjust based on your needs)
                corner_size = 10  # pixels
                
                # Check if contour is in any of the four corners
                is_in_corner = False
                
                # Top-left corner check
                if x <= rx1 + corner_size and y <= ry1 + corner_size:
                    is_in_corner = True
                # Top-right corner check  
                elif x + w >= rx2 - corner_size and y <= ry1 + corner_size:
                    is_in_corner = True
                # Bottom-left corner check
                elif x <= rx1 + corner_size and y + h >= ry2 - corner_size:
                    is_in_corner = True
                # Bottom-right corner check
                elif x + w >= rx2 - corner_size and y + h >= ry2 - corner_size:
                    is_in_corner = True
                
                if is_in_corner:
                    cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
                
                break  # No need to check other rectangles
            
        if inTheRect: continue  # Skip this contour entirely
        
        
        if w > 25 or h > 25:  # filter small noise
            parent = hierarchy[0][i][3]
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
            
            if parent != -1:
                continue  # skip child contours
                
            count += 1
         
            classification, start_point, end_point, approx = classify_contour(
                contours, hierarchy, i
            )
            print(f"Contour {count}: {classification}, Start: {start_point}, End: {end_point}")
            x, y, w, h = cv2.boundingRect(approx)
            # Draw bounding box
            cv2.rectangle(contour_img, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(contour_img, str(count) + " " + str(classification), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # # Draw approx points
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            contours_info.append({
                "type": classification,
                "start" : start_point, 
                "end": end_point,
                "approx": approx,
            })
            
            for pt in approx:
                cv2.circle(contour_img, tuple(pt[0]), 1, (255, 0, 0), -1)

            # Save info for merging - ONLY CHANGE: added has_arrow_head flag
         
    
    return contours_info, hierarchy, detectedContours, contour_img, detectText



# ---------------- Step 7: Visualization ----------------
def visualize(vertical_lines, horizontal_lines, image_rect):
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,3,1)
    plt.title("Image 3")
    plt.imshow(vertical_lines, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.title("Image 2")
    plt.imshow(horizontal_lines, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.title("Image 3")
    plt.imshow(cv2.cvtColor(image_rect, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def downloadImage(image):
    # cv2.imwrite("./images/atm.png", image)
    return None


# ---------------- Main workflow ----------------
def main(image_path):
    image, gray = load_image(image_path)
    thresh = threshold_image(gray)
    vertical_lines, horizontal_lines = extract_lines(thresh)
    
    # get contours to all detected vertical and horizontal lines
    vertical_segments = get_segments(vertical_lines)
    horizontal_segments = get_segments(horizontal_lines)
    
    # pairs those have almost identical vertical/horizontal lines, and get those lines that has no pairs
    vertical_pairs = find_vertical_pairs(vertical_segments)
    horizontal_pairs = find_horizontal_pairs(horizontal_segments)
    
    # get possible rectangles based on two pairs of vertical and horizontal lines
    rectangles, lines_to_remove = detect_rectangles(vertical_pairs, horizontal_pairs, vertical_segments, horizontal_segments)
    # remove/draw rectangles in the image
    image_rect = remove_rectangles(image, lines_to_remove, vertical_segments, horizontal_segments)
    
    # ==================== Preprocess again the image that after removing teh rect =============================== 
    noRectImg = image_rect.copy()      #copy image after removing rectangles
    # threshImg = threshold_image(cv2.cvtColor(noRectImg, cv2.COLOR_BGR2GRAY))    #threshold again
    
    contours, hierarchy, detectedContours, drawContour, textImg = detect_contours(noRectImg, rectangles)
    # downloadImage(textImg)
    image2 = image.copy()
    
    print("Contours: ", len(contours))
    
    # --- 3. OCR + bounding box info ---
    data = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    d1 = pytesseract.image_to_string(textImg)
    # --- 4. Convert grayscale to BGR for drawing ---
    textImg = cv2.cvtColor(textImg, cv2.COLOR_GRAY2BGR)

    # --- 5. Draw bounding boxes ---
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])

        if text != "":  # filter low confidence or empty
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            cv2.rectangle(textImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(textImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
   
    # visualize(thresh, drawContour, image2)
    visualize(image, drawContour, textImg)

# ---------------- Run ----------------
main("./images/std4.png")
