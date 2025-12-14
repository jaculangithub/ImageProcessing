import cv2
import numpy as np
import matplotlib.pyplot as plt
import math 
import pytesseract
from pytesseract import Output
import json

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
        vx, vy, vw, vh = vLine
        
        if vw < 10: continue
        
        for hLine in hLines:
            hx, hy, hw, hh = hLine
            
            if hh < 10: continue

            # Check for intersection
            if (vx < hx + hw and vx + vw > hx and
                vy < hy + hh and vy + vh > hy):
                print("Moern inters")
                diameter = min(vh, hw) 
                radius = diameter // 2
                
                if(vh < hw):
                    image_rect[vy + (vh // 2) - 5: vy + (vh // 2) + 5, vx + (vw//2) + radius] = bg_color      #remove right
                    image_rect[vy + (vh // 2) - 5: vy + (vh // 2) + 5, vx + (vw//2) - radius] = bg_color      #remove left
                else :
                    image_rect[hy + (hh // 2) - radius, hx + ( hw//2 ) - 5: hx + (hw//2) + 5] = bg_color
                    image_rect[hy + (hh // 2) + radius, hx + ( hw//2 ) - 5: hx + (hw//2) + 5] = bg_color
    

    # for vLine in vLines:
    #     x, y, w, h = vLine
        
    #     # For vertical lines: width should be much smaller than height
    #     aspect_ratio = w / h
    #     if aspect_ratio > 0.7:  # This is a proper vertical line
    #         middle_y = y + h // 2
            
    #         image_rect[middle_y-3:middle_y+3, x-3] = bg_color       #remove left 
    #         image_rect[middle_y-3:middle_y+3,x+w+3] = bg_color      #remove right
        
    for hLine in hLines:
        x, y, w, h = hLine
        
        # For horizontal lines: height should be much smaller than width
        aspect_ratio = h / w
        if aspect_ratio > 0.7:  # This is a proper horizontal line
         
            middle_x = x + w // 2
                    
            image_rect[y-3, middle_x-3:middle_x+3] = bg_color   #remove top
            image_rect[y+h+3, middle_x-3:middle_x+3] = bg_color #remove bottom
    
    return image_rect


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
    # dist, mx, mn = None, None, None
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
        
        dist, mx, mn = dynamic_tolerance, max(distances), min(distances)
        
        #  unlikely to be an arrow, it could be a different shape with multiple points
        if(dist < 5): return None, None
        
        if max(distances) - min(distances) <= dynamic_tolerance+5:
            seq_np = np.array(seq_points, dtype=np.int32)
            hull = cv2.convexHull(seq_np, returnPoints=True)
            hull_count = len(hull)
      
            # if hull_count >= 4:
            #     return seq_points, hull_count
            if hull_count == 3:
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
        x, y, w, h = cv2.boundingRect(child_cnt)
        child_epsilon = 0.08 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
        
        # If child has 4 points (diamond shape)
        if len(child_approx) == 4 and w/h > .9:
            print("Found diamond child - directed association")
            return "diamond node", None, None, approx
        
        # Move to next sibling child
        child_idx = hierarchy[0][child_idx][0]
    
    # Default for contours with children but no specific shape
    return "line", None, None, approx
    
    
def classify_contour(contours, hierarchy, idx):
    
    global epsilon_ratio
    
    child_idx = hierarchy[0][idx][2]
    
    if child_idx != -1: # has child, likely relationship
        print("has child")
        
        x, y, w, h = cv2.boundingRect(contours[child_idx])
        
        if (w > 10 and h > 10):
            # return None, None, None, None
            return classify_based_on_child(contours, hierarchy, idx)

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
    # detectText = threshImg.copy()
    
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

            # Draw approx points
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
                
         
    return contours_info, hierarchy, detectedContours, contour_img, detectText

# def detect_contours(image, rectangles, bg_color=0, min_area=100):
#     # Create working copies
#     working_image = image.copy()
#     gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
#     threshImg = threshold_image(gray)
    
#     detectedContours = threshImg.copy()
#     detectText = threshImg.copy()  # This will be our final text detection image
    
#     # First pass: find and remove diamond contours
#     contours, hierarchy = cv2.findContours(
#         detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#     ) 
    
#     for i, cnt in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(cnt)

#         # Traverse all children
#         child_idx = hierarchy[0][i][2]
#         while child_idx != -1:
#             child_cnt = contours[child_idx]
#             cx, cy, cw, ch = cv2.boundingRect(child_cnt)

#             if cw > 25 and ch > 25:
#                 epsilon = 0.08 * cv2.arcLength(child_cnt, True)
#                 approx = cv2.approxPolyDP(child_cnt, epsilon, True)

#                 # Check if it's inside any rectangle and in a corner
#                 inTheRect = False
#                 is_in_corner = False
#                 for rect in rectangles:
#                     rx1, ry1, rx2, ry2 = rect
#                     if cx >= rx1 and cy >= ry1 and (cx + cw) <= rx2 and (cy + ch) <= ry2:
#                         inTheRect = True
#                         corner_size = 10
#                         if (
#                             (cx <= rx1 + corner_size and cy <= ry1 + corner_size) or
#                             (cx + cw >= rx2 - corner_size and cy <= ry1 + corner_size) or
#                             (cx <= rx1 + corner_size and cy + ch >= ry2 - corner_size) or
#                             (cx + cw >= rx2 - corner_size and cy + ch >= ry2 - corner_size)
#                         ):
#                             is_in_corner = True
#                         break

#                 if inTheRect and is_in_corner:
#                     # Remove from ALL image versions consistently
#                     cv2.drawContours(detectText, [cnt], -1, bg_color, -1)
#                     cv2.drawContours(working_image, [cnt], -1, (255, 255, 255), -1)  # Also remove from working image

#                 # Diamond shape processing
#                 if len(approx) == 4:
#                     middle_y = cy + ch // 2
#                     middle_x = cx + cw // 2

#                     # Remove from ALL image versions
#                     for img in [working_image, detectText]:
#                         img[middle_y-3:middle_y+3, cx-4] = (255, 255, 255) if len(img.shape) == 3 else 255
#                         img[middle_y-3:middle_y+3, cx+cw+4] = (255, 255, 255) if len(img.shape) == 3 else 255
#                         img[cy-4, middle_x-3:middle_x+3] = (255, 255, 255) if len(img.shape) == 3 else 255
#                         img[cy+ch+4, middle_x-3:middle_x+3] = (255, 255, 255) if len(img.shape) == 3 else 255

#             child_idx = hierarchy[0][child_idx][0]

#     # Second pass: regenerate threshold from modified working image
#     gray_modified = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
#     threshImg_modified = threshold_image(gray_modified)
    
#     detectedContours_modified = threshImg_modified.copy()
    
#     # Find contours on the modified image
#     contours, hierarchy = cv2.findContours(
#         detectedContours_modified, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
#     )

#     contour_img = cv2.cvtColor(detectedContours_modified, cv2.COLOR_GRAY2BGR)
#     contours_info = []
#     count = 0
    
#     for i, cnt in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(cnt)
        
#         # Check if contour is inside any rectangle
#         inTheRect = False
#         for rect in rectangles:
#             rx1, ry1, rx2, ry2 = rect
#             if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
#                 inTheRect = True
                
#                 # Check if in corner
#                 corner_size = 10
#                 is_in_corner = (
#                     (x <= rx1 + corner_size and y <= ry1 + corner_size) or
#                     (x + w >= rx2 - corner_size and y <= ry1 + corner_size) or
#                     (x <= rx1 + corner_size and y + h >= ry2 - corner_size) or
#                     (x + w >= rx2 - corner_size and y + h >= ry2 - corner_size)
#                 )
                
#                 if is_in_corner:
#                     # Remove from detectText (our final output)
#                     cv2.drawContours(detectText, [cnt], -1, bg_color, -1)
#                     # Also remove from working image for consistency
#                     cv2.drawContours(working_image, [cnt], -1, (255, 255, 255), -1)
                
#                 break
        
#         if inTheRect: 
#             continue
        
#         if w > 25 or h > 25:
#             parent = hierarchy[0][i][3]
            
#             # Remove ALL internal contours from detectText
#             cv2.drawContours(detectText, [cnt], -1, bg_color, -1)
            
#             if parent != -1:
#                 continue
            
#             count += 1
            
#             classification, start_point, end_point, approx = classify_contour(
#                 contours, hierarchy, i
#             )
#             print(f"Contour {count}: {classification}, Start: {start_point}, End: {end_point}")
#             x, y, w, h = cv2.boundingRect(approx)
         
#             # Draw on contour_img for visualization only
#             cv2.rectangle(contour_img, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 1)
#             cv2.putText(contour_img, str(count) + " " + str(classification), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5, (0, 0, 255), 1, cv2.LINE_AA)

#             epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, epsilon, True)

#             contours_info.append({
#                 "type": classification,
#                 "start": start_point, 
#                 "end": end_point,
#                 "approx": approx,
#             })
            
#             for pt in approx:
#                 cv2.circle(contour_img, tuple(pt[0]), 1, (255, 0, 0), -1)
                
#     return contours_info, hierarchy, detectedContours_modified, contour_img, detectText

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

def merge_nearby_texts(texts, x_margin=10, y_margin=10, dash_width_threshold=5, align_threshold=5, min_vertical_group=5):
    if not texts:
        return []

    # --- STEP 1: Remove dash noise ---
    dash_candidates = [t for t in texts if t[2] < dash_width_threshold]
    remove_set = set()
    
    for i, entry in enumerate(dash_candidates):
        x_i = entry[0]
        aligned_group = [entry]
        for other in dash_candidates:
            if other is not entry and abs(other[0] - x_i) <= align_threshold:
                aligned_group.append(other)

        if len(aligned_group) >= min_vertical_group:
            for e in aligned_group:
                remove_set.add(e)

    texts = [t for t in texts if t not in remove_set]

    # --- STEP 2: Better line detection ---
    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))
    
    # Improved line grouping using bounding box overlap
    lines = []
    current_line = [texts_sorted[0]]
    
    for i in range(1, len(texts_sorted)):
        current_text = texts_sorted[i]
        last_in_line = current_line[-1]
        
        # Use actual bounding box overlap for line detection
        current_top, current_bottom = current_text[1], current_text[1] + current_text[3]
        last_top, last_bottom = last_in_line[1], last_in_line[1] + last_in_line[3]
        
        # Check for significant vertical overlap
        vertical_overlap = min(current_bottom, last_bottom) - max(current_top, last_top)
        same_line = vertical_overlap > 0 or abs(current_top - last_top) <= y_margin
        
        if same_line:
            current_line.append(current_text)
        else:
            lines.append(current_line)
            current_line = [current_text]
    
    if current_line:
        lines.append(current_line)

    # --- STEP 3: Conservative horizontal merging ---
    merged_horizontal = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[0])
        current_group = [line_sorted[0]]
        
        for text in line_sorted[1:]:
            last = current_group[-1]
            last_right = last[0] + last[2]
            current_left = text[0]
            
            # Strict horizontal gap check
            horizontal_gap = current_left - last_right
            
            if horizontal_gap <= x_margin:
                current_group.append(text)
            else:
                # Merge current group
                if current_group:
                    x0 = min(t[0] for t in current_group)
                    y0 = min(t[1] for t in current_group)
                    x1 = max(t[0] + t[2] for t in current_group)
                    y1 = max(t[1] + t[3] for t in current_group)
                    sentence = " ".join(t[4] for t in current_group)
                    merged_horizontal.append((x0, y0, x1 - x0, y1 - y0, sentence))
                current_group = [text]
        
        if current_group:
            x0 = min(t[0] for t in current_group)
            y0 = min(t[1] for t in current_group)
            x1 = max(t[0] + t[2] for t in current_group)
            y1 = max(t[1] + t[3] for t in current_group)
            sentence = " ".join(t[4] for t in current_group)
            merged_horizontal.append((x0, y0, x1 - x0, y1 - y0, sentence))

    # --- STEP 4: Very conservative vertical merging (optional) ---
    # Only merge if you're sure you want multi-line text combined
    final_merged = merged_horizontal  # Skip vertical merging if it's causing issues
    
    # Or use very strict vertical merging:
    # final_merged = conservative_vertical_merge(merged_horizontal, y_margin)
    
    return final_merged

def conservative_vertical_merge(texts, y_margin):
    """Only merge text that are clearly part of the same paragraph"""
    texts_sorted = sorted(texts, key=lambda t: t[1])
    final = []
    used = set()
    
    for i, text1 in enumerate(texts_sorted):
        if i in used:
            continue
            
        group = [text1]
        used.add(i)
        x1, y1, w1, h1, _ = text1
        
        for j, text2 in enumerate(texts_sorted):
            if j in used:
                continue
                
            x2, y2, w2, h2, _ = text2
            
            # Very strict conditions for vertical merging:
            is_directly_below = y2 >= y1 + h1  # text2 is below text1
            vertical_proximity = (y2 - (y1 + h1)) <= y_margin  # Close vertically
            horizontal_alignment = abs((x1 + w1/2) - (x2 + w2/2)) <= w1 * 0.3  # Similar horizontal center
            similar_width = abs(w1 - w2) <= w1 * 0.4  # Similar width
            
            if (is_directly_below and vertical_proximity and 
                horizontal_alignment and similar_width):
                group.append(text2)
                used.add(j)
        
        if len(group) > 1:
            # Merge the group
            x0 = min(t[0] for t in group)
            y0 = min(t[1] for t in group)
            x1 = max(t[0] + t[2] for t in group)
            y1 = max(t[1] + t[3] for t in group)
            sentence = " ".join(t[4] for t in group)
            final.append((x0, y0, x1 - x0, y1 - y0, sentence))
        else:
            final.append(text1)
    
    return final


def processText(texts, y_threshold=5):
    """
    Sort texts: group by lines (similar Y), then sort each line left to right
    """
    if not texts:
        return ""
    
    # Sort all texts by Y first, then X
    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))
    
    # Group by lines based on Y threshold
    lines = []
    current_line = [texts_sorted[0]]
    
    for i in range(1, len(texts_sorted)):
        current_text = texts_sorted[i]
        prev_text = texts_sorted[i-1]
        
        # Calculate Y centers
        current_center_y = current_text[1] + current_text[3] / 2
        prev_center_y = prev_text[1] + prev_text[3] / 2
        
        # Check if Y difference is within threshold (same line)
        if abs(current_center_y - prev_center_y) <= y_threshold:
            current_line.append(current_text)
        else:
            # Sort current line left to right and add to lines
            current_line.sort(key=lambda t: t[0])
            lines.append(current_line)
            current_line = [current_text]
    
    # Add the last line
    if current_line:
        current_line.sort(key=lambda t: t[0])
        lines.append(current_line)
    
    # Build the final text
    words = ""
    for line in lines:
        for text in line:
            _, _, _, _, txt = text
            words = words + " " + txt
    
    return words.strip()


def mergeTransition(transitions, texts, margin=5):
    # Standardize point formats to numpy arrays
    def standardize_point(point):
        if point is None:
            return None
        if isinstance(point, np.ndarray):
            point = point.astype(np.int32)  # Use integers instead of floats
            if point.ndim > 1:
                point = point.flatten()
            return point
        elif isinstance(point, tuple):
            # Convert tuple of numpy scalars or regular tuples to numpy array
            if all(isinstance(x, (np.integer, np.floating)) for x in point):
                return np.array([float(x) for x in point], dtype=np.float64)
            else:
                return np.array(point, dtype=np.float64)
        elif hasattr(point, '__iter__'):
            return np.array(point, dtype=np.float64)
        else:
            return np.array([point], dtype=np.float64)

    # detect nearby text
    for transition in transitions:
        # Get bounding box of transition
        tx1, ty1, tw, th = cv2.boundingRect(transition["approx"])
        tx2, ty2 = tx1 + tw, ty1 + th
        
        # Expand transition bbox with margin
        trans_bbox_expanded = (tx1 - margin, ty1 - margin, tx2 + margin, ty2 + margin)
        ex1, ey1, ex2, ey2 = trans_bbox_expanded
        
        # Find texts whose bbox intersects with expanded transition bbox
        label = ""
        for text in texts:
            bx, by, bw, bh, text_content = text
            text_bbox = (bx, by, bx + bw, by + bh)
            bx1, by1, bx2, by2 = text_bbox
            
            # Check if text bbox intersects with expanded transition bbox
            if (bx1 < ex2 and bx2 > ex1 and by1 < ey2 and by2 > ey1):
                label = text_content
                transition["label"] = label
                break
        
    # merge transitions that has same label
    removeTransitions = []
    mergeTransitions = []
    
    for i in range(len(transitions)):
        if transitions[i].get("label") is None:
            continue
        for j in range(i + 1, len(transitions)):
            if transitions[j].get("label") is None or (transitions[i].get("end") is not None and transitions[j].get("end") is not None):
                continue
            if transitions[i].get("label") == transitions[j].get("label"):
                label = transitions[i]["label"]
                print(f"Transition {i} and {j} have same label: '{label}'")
                
                arrowhead = None
                startPoint = None
                
                iPoint1, iPoint2 = None, None
                if(transitions[i].get("start") is None):
                    iPoint1, iPoint2, _ = find_farthest_points(transitions[i].get("approx"))
                else:
                    iPoint1, iPoint2 = transitions[i].get("start"), transitions[i].get("end")
                    arrowhead = iPoint2
                    
                jPoint1, jPoint2 = None, None
                if(transitions[j].get("start") is None):
                    jPoint1, jPoint2, _ = find_farthest_points(transitions[j].get("approx"))
                else:
                    jPoint1, jPoint2 = transitions[j].get("start"), transitions[j].get("end")
                    arrowhead = jPoint2
                
                # Standardize points to numpy arrays
                iPoint1 = standardize_point(iPoint1)
                iPoint2 = standardize_point(iPoint2)
                jPoint1 = standardize_point(jPoint1)
                jPoint2 = standardize_point(jPoint2)
                
                if arrowhead is not None:  # one of the transition has arrowhead
                    transition_type = "directed arrow"
                    
                    # Determine which transition has the arrowhead
                    if transitions[i].get("end") is not None:
                        # i has arrowhead, j is plain line
                        arrowhead = standardize_point(transitions[i]["end"])
                        # For plain line j, find the point farthest from the arrowhead
                        points = transitions[j].get("approx")
                        max_distance = 0
                        for point in points:
                            point_arr = standardize_point(point)
                            distance = np.linalg.norm(arrowhead - point_arr)
                            if distance > max_distance:
                                max_distance = distance
                                startPoint = point_arr  # Store as numpy array
                    else:
                        # j has arrowhead, i is plain line
                        arrowhead = standardize_point(transitions[j]["end"])
                        # For plain line i, find the point farthest from the arrowhead
                        points = transitions[i].get("approx")
                        max_distance = 0
                        for point in points:
                            point_arr = standardize_point(point)
                            distance = np.linalg.norm(arrowhead - point_arr)
                            if distance > max_distance:
                                max_distance = distance
                                startPoint = point_arr  # Store as numpy array
                
                else:  # both transitions are plain lines
                    transition_type = "line"
                    # Combine both contours and find the two farthest points
                    combined_approx = np.vstack([transitions[i].get("approx"), transitions[j].get("approx")])
                    start, end, _ = find_farthest_points(combined_approx)
                    startPoint = standardize_point(start)
                    arrowhead = standardize_point(end)
                
                # Merge the contours
                merged_approx = np.vstack([transitions[i].get("approx"), transitions[j].get("approx")])
                
                # Mark transitions for removal
                removeTransitions.append(transitions[i])
                removeTransitions.append(transitions[j])
                
                # Ensure startPoint and arrowhead are properly standardized
                startPoint = standardize_point(startPoint)
                arrowhead = standardize_point(arrowhead)
                
                print("Start: ", startPoint)
                print("end: ", arrowhead)  
                
                # Create merged transition
                mergeTransitions.append({
                    "type": transition_type,
                    "start": startPoint,
                    "end": arrowhead,
                    "approx": merged_approx,
                    "label": transitions[i]["label"]
                })
                
                break  # break inner loop after merging
    
    # Remove the original transitions
    for transition in removeTransitions:
        if transition in transitions:
            transitions.remove(transition)
    
    # Add the merged transitions
    for transition in mergeTransitions:
        transitions.append(transition)
    
    return transitions


def getNodeID(props, point, pos, transition, margin = 8):
    
    # if all detected as plain line
    if(props[0] == "plainLine"):
        
        decisionNodes = [props[1] for i in props[1]["type"] == "diamond"]
        
        return
    
    # [action, othernotations, label, text_arr]
    allNodes = [props[0], props[1]]
    
    # for all arrow line
    for nodes in allNodes:
        for node in nodes:
            x, y, w, h = node["bbox"]
            px, py = point[0], point[1]
            
            if((x-margin <= px) and (x+w+margin >= px) and (y-margin <= py) and (y+h+margin > py)):
                # inside thee action
                node["hasConnection"] = True
                
                if pos not in node:
                    node[pos] = []
                
                node[pos].append({
                    "transitionID": transition.get("id")
                })
                return node["id"]
            
            
            if node["type"] == "diamond":
                allTextLabel = props[3]  # assuming props[3] = list of text labels
                for textLabel in allTextLabel:
                    tx, ty, tw, th, text = textLabel

                    # Calculate bounding box edges
                    node_left, node_top, node_right, node_bottom = x, y, x + w, y + h
                    text_left, text_top, text_right, text_bottom = tx, ty, tx + tw, ty + th

                    # Compute actual distance between boxes
                    margin = 30
                    
                    # Calculate horizontal and vertical distances
                    horizontal_dist = max(0, node_left - text_right, text_left - node_right)
                    vertical_dist = max(0, node_top - text_bottom, text_top - node_bottom)
                    
                    # Calculate Euclidean distance between the closest points
                    distance = (horizontal_dist**2 + vertical_dist**2)**0.5
                    
                    if distance <= margin and props[2] == text:
                        print("Text near diamond node:", text)
                        
                        node["hasConnection"] = True
                        
                        if pos not in node:
                            node[pos] = []
                    
                        node[pos].append({
                            "transitionID": transition.get("id")
                        })
                        
                        return node["id"]

    return None


def restructureData(rectangles, contours, texts):

    # find the text inside the rectangles
    actionNodes = []
    transitions = []
    otherNotations = []
    
    # find actions nodes
    for rect in rectangles:
        rx, ry, rw, rh = rect
        actionTexts = []
        
        for text in texts:
            tx, ty, tw, th, word = text         
            if((rx < tx) and (tx + tw < rw) and (ry < ty) and (rh > (ty + th))):
                actionTexts.append(text)
                
        txt = processText(actionTexts)
        print("Text: ", txt)
        actionNodes.append({
            "id": "act" + str(len(actionNodes)+1),
            "type": "ActionNode",
            "bbox": (rx, ry, rw - rx, rh - ry),  # Store as (x, y, w, h)
            "label": txt
        })
        
        texts = [text for text in texts if text not in actionTexts]
    
    # merge text
    text_merged = merge_nearby_texts(texts)
    
    # separate other notations to arrow
    for contour in contours:          
        if(contour["type"] in ["end node", "destruction node", "start node", "diamond node"]):
            classification = contour["type"].split()[0]
            bbox = cv2.boundingRect(contour["approx"])
            print("Type :", classification, "node")                    
            otherNotations.append({
                "id": "notation" + str(len(otherNotations)+1),
                "type": classification,
                "bbox": bbox,
            })

        elif (contour["type"] in ["line", "directed arrow"]):
            transitions.append(contour)
        
    
    print(text_merged) 
    # print(len(transitions))
    # merge transitions, arrow/lines
    print("Old n = ", len(transitions))
    transitions = mergeTransition(transitions, text_merged)
    
    # adding id for each transition
    for i, transition in enumerate(transitions): 
        transition["id"] = "line" + str(i + 1)
         
    
    print(len(transitions))
    print(len(actionNodes))
    print(len(otherNotations))
    
    # restructure data
    c = 0
    
    for transition in transitions:
        c += 1
        if (transition.get("type") == "directed arrow"):
            start, end = transition.get("start"), transition.get("end")
            
            props = (actionNodes, otherNotations, transition.get("label"), text_merged)
            
            startingNodeID = getNodeID(props, start, "from", transition)
            destinationNodeID = getNodeID(props, end, "to", transition)    

            print(
                "start: ", startingNodeID,
                "end: ", destinationNodeID
            )
        
        
    for actionNode in actionNodes:
        if(actionNode.get("hasConnection") is not None):
            print(
                actionNode["id"], 
                actionNode["hasConnection"],
                actionNode["bbox"], 
                " From ", len(actionNode.get("from", [])), 
                " TO ", len(actionNode.get("to", []))
            )
    
    structuredData = {
        "nodes": [],
        "transitions": []
    }
    
    for node in actionNodes:
        structuredData["nodes"].append({
            "id": node.get("id"),
            "type": node.get("type"),
            "bbox": [int(x) for x in node["bbox"]],
            "label": node.get("label"),
            "from": node.get("from", None),
            "to": node.get("to", None),
        })
    
    for notation in otherNotations:
        structuredData["nodes"].append({
            "id": notation["id"],
            "type": notation["type"],
            "bbox": [int(x) for x in notation["bbox"]],
            "from": notation.get("from", None),
            "to": notation.get("to", None)
        })
    
    for transition in transitions:
        transition_data = {
            "id": transition["id"],
            "type": transition["type"],
            "start": [int(transition["start"][0]), int(transition["start"][1])],
            "end": [int(transition["end"][0]), int(transition["end"][1])]
        }
        if "label" in transition:
            transition_data["label"] = transition["label"]
            
        structuredData["transitions"].append(transition_data)
    
    return structuredData 


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
    txtM = textImg.copy()
    print("Contours: ", len(contours))
    
    # --- 3. OCR + bounding box info ---
    textData = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    
    # --- 4. Convert grayscale to BGR for drawing ---
    textImg = cv2.cvtColor(textImg, cv2.COLOR_GRAY2BGR)

    # --- 5. Draw bounding boxes ---
    n_boxes = len(textData['level'])
    for i in range(n_boxes):
        text = textData['text'][i].strip()
        conf = int(textData['conf'][i])

        if text != "":  # filter low confidence or empty
            (x, y, w, h) = (textData['left'][i], textData['top'][i], textData['width'][i], textData['height'][i])
            cv2.rectangle(textImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(textImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    texts = []
    for i in range(len(textData['text'])):
        text = textData['text'][i].strip()
        if text != "":  # ignore empty OCR outputs
            x = int(textData['left'][i])
            y = int(textData['top'][i])
            w = int(textData['width'][i])
            h = int(textData['height'][i])
            texts.append((x, y, w, h, text))
    
    structuredData = restructureData(rectangles, contours, texts)
    
    print(json.dumps(structuredData, indent=2))
    
    # visualize(thresh, drawContour, image2)
    visualize(vertical_lines, horizontal_lines, textImg)

# ---------------- Run ----------------
main("./images/actWithMultipleForkJoin.png")

