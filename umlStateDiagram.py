import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json
import time

epsilon_ratio = 0.04

# ---------------- Step 0: Load image ----------------
def load_image_from_file(file):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# ---------------- Step 1: Threshold ----------------
def threshold_image(gray, thresh_val=100):
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
    c = 0
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
                minHWidth = min(h1[2], h2[2])
                minVHeight = min(v1[3], v2[3])
                c = c + 1
                cornerRadius = min(minHWidth, minVHeight) * 0.3
                
                width_tolerance_ratio = 0.3
                height_tolerance_ratio = 0.3
                
                if((maxWidth - h1Width) < cornerRadius * 2 and (maxHeight - v1Height) < cornerRadius * 2):
                    
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
                diameter = min(vh, hw) 
                radius = diameter // 2
                
                if(vh < hw):
                    image_rect[vy + (vh // 2) - 5: vy + (vh // 2) + 5, vx + (vw//2) + radius + 1] = bg_color      #remove right
                    image_rect[vy + (vh // 2) - 5: vy + (vh // 2) + 5, vx + (vw//2) - radius - 1] = bg_color      #remove left
                else :
                    image_rect[hy + (hh // 2) - radius - 1, hx + ( hw//2 ) - 5: hx + (hw//2) + 5] = bg_color
                    image_rect[hy + (hh // 2) + radius + 1, hx + ( hw//2 ) - 5: hx + (hw//2) + 5] = bg_color
    
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
    
    # The point in the sequence is the end (diamond side)
    if point1_in_seq and not point2_in_seq:
        return point2, point1  # point1 is end
    elif point2_in_seq and not point1_in_seq:
        return point1, point2  # point2 is end
    else:
        # Both or neither in sequence, return as plain association
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
        
        # unlikely an arrow, it could be a different shape with multiple points
        if(dynamic_tolerance < 5): return None, None
        
        if max(distances) - min(distances) <= dynamic_tolerance+5:
            seq_np = np.array(seq_points, dtype=np.int32)
            hull = cv2.convexHull(seq_np, returnPoints=True)
            hull_count = len(hull)
      
            if hull_count == 3:
                return seq_points, hull_count
    
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
            return "diamond node", None, None, approx
        
        # Move to next sibling child
        child_idx = hierarchy[0][child_idx][0]
    
    # Default for contours with children but no specific shape
    return "line", None, None, approx

def classify_contour(contours, hierarchy, idx):
    
    global epsilon_ratio
    
    child_idx = hierarchy[0][idx][2]
    
    if child_idx != -1: # has child, likely relationship
        x, y, w, h = cv2.boundingRect(contours[child_idx])
        
        if (w > 10 and h > 10):
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
    
    # Check for circular shape (start node)
    if bbox_area > 0 and (area / bbox_area) >= 0.7 and circularity >= 0.7:
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
            break
        
        elif epsilon_ratio <= 0.005:
            break

        epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx

    if has_five_seq:
        if convexHull and convexHull == 3:
            classification = "directed arrow"
            start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
            
            return classification, start_point, end_point, approx
        else: 
            classification = "line" 
            return "line", None, None, approx
    
    return "line", None, None, approx

def detect_contours(image, rectangles, bg_color=0, min_area=100):

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
        is_in_corner = False
        inCompositeState = False
        
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            # Check if contour bbox is within rectangle bbox
            if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                
                if(rx2 - rx1) > 200 or (ry2 - ry1) < 200: 
                    inCompositeState = True
                
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
        
        # Skip contours inside composite state
        if inTheRect: 
            # check if the contour is in the composite state
            if is_in_corner:
                continue  # Skip this contour entirely
            
            if not inCompositeState:
                for rect in rectangles:
                    rx1, ry1, rx2, ry2 = rect
                    if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                        if(rx2 - rx1) > 200 or (ry2 - ry1) < 200: 
                            inCompositeState = True
                
                if not inCompositeState:
                    continue  # Skip this contour entirely
        
        if w > 25 or h > 25:  # filter small noise
            parent = hierarchy[0][i][3]
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
            
            if parent != -1:
                continue  # skip child contours
                
            count += 1
            
            classification, start_point, end_point, approx = classify_contour(
                contours, hierarchy, i
            )
            x, y, w, h = cv2.boundingRect(approx)
         
            contours_info.append({
                "type": classification,
                "start" : start_point, 
                "end": end_point,
                "approx": approx,
            })
         
    return contours_info, hierarchy, detectedContours, contour_img, detectText

def merge_nearby_texts(texts, x_margin=15, y_margin=20, dash_width_threshold=5, align_threshold=5, min_vertical_group=5):
    """
    Merge nearby text boxes into sentences while ignoring vertical dash-like noise.
    """
    if not texts:
        return []

    # --- STEP 1: Detect and remove vertical dash groups ---
    dash_candidates = [t for t in texts if t[2] < dash_width_threshold]

    remove_set = set()
    for i, entry in enumerate(dash_candidates):
        x_i = entry[0]
        aligned_group = [entry]
        for other in dash_candidates:
            if other is not entry:
                if abs(other[0] - x_i) <= align_threshold:
                    aligned_group.append(other)

        if len(aligned_group) >= min_vertical_group:
            for e in aligned_group:
                remove_set.add(e)

    texts = [t for t in texts if t not in remove_set]

    # --- STEP 2: Group by horizontal lines first ---
    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))

    # First pass: merge horizontally on same line
    horizontal_lines = []
    current_line = [texts_sorted[0]]

    for i in range(1, len(texts_sorted)):
        current_text = texts_sorted[i]
        last_text = texts_sorted[i-1]
        
        x1, y1, w1, h1, _ = current_text
        x2, y2, w2, h2, _ = last_text
        
        # Check if same line (similar Y position)
        same_line = abs((y1 + h1/2) - (y2 + h2/2)) <= y_margin / 2
        
        if same_line:
            current_line.append(current_text)
        else:
            horizontal_lines.append(current_line)
            current_line = [current_text]

    if current_line:
        horizontal_lines.append(current_line)

    # Merge each horizontal line
    merged_horizontal = []
    for line in horizontal_lines:
        line_sorted = sorted(line, key=lambda t: t[0])  # Sort by X
        current_group = [line_sorted[0]]
        
        for text in line_sorted[1:]:
            last_x, last_y, last_w, last_h, last_text = current_group[-1]
            x, y, w, h, current_text = text
            
            # Check horizontal proximity
            horizontal_gap = x - (last_x + last_w)
            
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
        
        # Merge remaining group
        if current_group:
            x0 = min(t[0] for t in current_group)
            y0 = min(t[1] for t in current_group)
            x1 = max(t[0] + t[2] for t in current_group)
            y1 = max(t[1] + t[3] for t in current_group)
            sentence = " ".join(t[4] for t in current_group)
            merged_horizontal.append((x0, y0, x1 - x0, y1 - y0, sentence))

    # --- STEP 3: Group vertically for multi-line text ---
    merged_horizontal_sorted = sorted(merged_horizontal, key=lambda t: (t[1], t[0]))
    
    vertical_groups = []
    used = set()
    
    for i, text1 in enumerate(merged_horizontal_sorted):
        if i in used:
            continue
            
        vertical_group = [text1]
        used.add(i)
        
        x1, y1, w1, h1, t1 = text1
        
        # Find vertically aligned text below
        for j, text2 in enumerate(merged_horizontal_sorted):
            if j in used:
                continue
                
            x2, y2, w2, h2, t2 = text2
            
            # Check if text2 is below text1 and vertically aligned
            is_below = y2 > y1 + h1  # text2 is below text1
            vertical_gap = y2 - (y1 + h1)
            horizontal_overlap = not (x1 + w1 < x2 or x2 + w2 < x1)
            similar_width = abs(w1 - w2) <= max(w1, w2) * 0.5  # Within 50% width difference
            
            # More flexible vertical grouping conditions
            if (is_below and vertical_gap <= y_margin * 2 and 
                (horizontal_overlap or similar_width)):
                vertical_group.append(text2)
                used.add(j)
        
        vertical_groups.append(vertical_group)

    # --- STEP 4: Merge vertical groups ---
    final_merged = []
    
    for group in vertical_groups:
        if len(group) == 1:
            # Single line, just add as is
            final_merged.append(group[0])
        else:
            # Multiple lines, merge with proper formatting
            group_sorted = sorted(group, key=lambda t: t[1])  # Sort by Y
            
            x0 = min(t[0] for t in group_sorted)
            y0 = min(t[1] for t in group_sorted)
            x1 = max(t[0] + t[2] for t in group_sorted)
            y1 = max(t[1] + t[3] for t in group_sorted)
            
            # Combine text with space (not newline for now)
            sentence = " ".join(t[4] for t in group_sorted)
            final_merged.append((x0, y0, x1 - x0, y1 - y0, sentence))

    return final_merged

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

def mergeTransition(transitions, texts, margin=10):
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

    text_to_remove = set()
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
                text_to_remove.add(text)
                break
        
    # Remove used texts
    texts[:] = [t for t in texts if t not in text_to_remove]
    
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

def getNodeID(props, point, pos, transition, texts, margin = 8):
    
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
                
                if(node.get("type") == "Composite State"):
                    startPoint = transition.get("start")
                    endPoint = transition.get("end")
                    
                    sx, sy = startPoint[0], startPoint[1]
                    ex, ey = endPoint[0], endPoint[1]
                    
                    if((x <= sx <= x+w) and (y <= sy <= y+h) and (x <= ex <= x+w) and (y <= ey <= y+h)):
                        continue
                
                node["hasConnection"] = True
                
                if pos not in node:
                    node[pos] = []
                
                node[pos].append({
                    "transitionID": transition.get("id")
                })
                return node["id"]
            
    # find nodes with nearby text label taht outside to its area
    for nodes in allNodes:
        for node in nodes:
            x, y, w, h = node["bbox"]
            px, py = point[0], point[1]
            
            for text in texts:
                tx, ty, tw, th, word = text         
                
                if (word != props[2]): #if not same label, this is not from the arrow
                    continue
                
                # Calculate bounding box edges
                node_left, node_top, node_right, node_bottom = x, y, x + w, y + h
                text_left, text_top, text_right, text_bottom = tx, ty, tx + tw, ty + th

                # calculate is near the node bbox
                horizontal_dist = max(0, node_left - text_right, text_left - node_right)
                vertical_dist = max(0, node_top - text_bottom, text_top - node_bottom)
                distance = (horizontal_dist**2 + vertical_dist**2)**0.5
                
                if(0 < vertical_dist <= 40 or 0 < horizontal_dist <= 40):
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
    stateNodes = []
    transitions = []
    otherNotations = []
    copy_texts = texts.copy()
    
    # determine the hierarchy of states, meaning I want to find the states of the composite state if have,
    # a composite state is a state/rect that has another rect inside it,
    # if dont have classify them as normal state
    
    for rect in rectangles:
        x1, y1, x2, y2 = rect

        for otherRect in rectangles:
            if otherRect is rect:
                continue
            ox1, oy1, ox2, oy2 = otherRect
            # check if the ox1 is inside of rect 
            if(x1 < ox1 and ox2 < x2 and y1 < oy1 and oy2 < y2):
                otherNotations.append({
                    "id": "notation" + str(len(otherNotations)+1),
                    "type": "Composite State",
                    "bbox": (x1, y1, x2 - x1, y2 - y1),  # Store as (x, y, w, h)
                    "label": ""
                })
                rectangles.remove(rect)
                break 
            
    # find actions nodes
    for rect in rectangles:
        rx, ry, rw, rh = rect
        actionTexts = []
        
        for text in texts:
            tx, ty, tw, th, word = text         
            if((rx < tx) and (tx + tw < rw) and (ry < ty) and (rh > (ty + th))):
                actionTexts.append(text)
                
        txt = processText(actionTexts)
        stateNodes.append({
            "id": "state" + str(len(stateNodes)+1),
            "type": "State",
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
            otherNotations.append({
                "id": "notation" + str(len(otherNotations)+1),
                "type": classification,
                "bbox": bbox,
            })

        elif (contour["type"] in ["line", "directed arrow"]):
            transitions.append(contour)

    transitions = mergeTransition(transitions, text_merged)
    
    # adding id for each transition
    for i, transition in enumerate(transitions): 
        transition["id"] = "line" + str(i + 1)
    
    # find composite state labels and its substates
    for otherNotation in otherNotations:  
        if otherNotation.get("type") is not None and otherNotation.get("type") == "Composite State":
            x, y, w, h = otherNotation["bbox"]
            
            # detect teh the text remaining that is inside the composite state, it is the composite state label/name
            for text in text_merged:
                
                tx, ty, tw, th, word = text         
                x, y, w, h = otherNotation["bbox"]
                
                if((x < tx) and (tx + tw < x + w) and (y < ty) and (y + h > (ty + th)) and 
                   (y + ty + th) / h < 0.3):  # only consider text in the upper 30% area as label
                    otherNotation["label"] = word
                    text_merged.remove(text)
                    break
           
            subStates = []
            # then detect what are those substates inside it
            for stateNode in stateNodes:
                sx, sy, sw, sh = stateNode["bbox"]
                if((x < sx) and (sx + sw < x + w) and (y < sy) and (y + h > (sy + sh))):
                    subStates.append(stateNode["id"])
                    stateNode["parent"] = otherNotation["id"]
            
            for innerNotation in otherNotations:
                if innerNotation == otherNotation:
                    continue
                
                ix, iy, iw, ih = innerNotation["bbox"]
                
                if((x < ix) and (ix + iw < x + w) and (y < iy) and (y + h > (iy + ih))):
                    subStates.append(innerNotation["id"])
                    innerNotation["parent"] = otherNotation["id"]

            otherNotation["subStates"] = subStates

    # restructure data
    for transition in transitions:
        if (transition.get("type") == "directed arrow"):
            start, end = transition.get("start"), transition.get("end")
            
            props = (stateNodes, otherNotations, transition.get("label"), text_merged)
            
            startingNodeID = getNodeID(props, start, "from", transition, copy_texts)
            destinationNodeID = getNodeID(props, end, "to", transition, copy_texts)    
            
            transition["startNode"] = startingNodeID
            transition["endNode"] = destinationNodeID

    structuredData = {
        "nodes": [],
        "transitions": []
    }
    
    for node in stateNodes:
        structuredData["nodes"].append({
            "id": node.get("id"),
            "type": node.get("type"),
            "label": node.get("label"),
            "bbox": [int(x) for x in node["bbox"]],
            "parent": node.get("parent", None),
            "from": node.get("from", []),
            "to": node.get("to", []),
        })
    
    for notation in otherNotations:
        structuredData["nodes"].append({
            "id": notation["id"],
            "type": notation["type"],
            "label": notation.get("label", ""),
            "bbox": [int(x) for x in notation["bbox"]],
            "parent": notation.get("parent", None),
            "subStates": notation.get("subStates", []),
            "from": notation.get("from", []),
            "to": notation.get("to", []),
        })
    
    for transition in transitions:
        transition_data = {
            "id": transition["id"],
            "type": transition["type"],
            "start": [int(transition["start"][0]), int(transition["start"][1])] if transition.get("start") is not None else None,
            "end": [int(transition["end"][0]), int(transition["end"][1])] if transition.get("end") is not None else None,
            "label": transition.get("label", ""),
            "startNode": transition.get("startNode"),
            "endNode": transition.get("endNode")
        }
      
        structuredData["transitions"].append(transition_data)
    
    return structuredData 

# ---------------- Main workflow ----------------
def process_state_diagram(file):
    start_time = time.time()
    
    image, gray = load_image_from_file(file)
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
    
    contours, hierarchy, detectedContours, drawContour, textImg = detect_contours(noRectImg, rectangles)
    
    # --- OCR + bounding box info ---
    textData = pytesseract.image_to_data(textImg, output_type=Output.DICT)

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
    
    end_time = time.time()
    
    executionTime = end_time - start_time
    
    structuredData["executionTime"] = f"{executionTime} second/s"
    
    return structuredData