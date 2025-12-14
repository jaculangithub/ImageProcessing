import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json
import time
import math

epsilon_ratio = 0.04

# ---------------- Step 0: Load image ----------------
def load_image_from_file(file):
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# ---------------- Step 1: Threshold ----------------
def threshold_image(gray, thresh_val=150):
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
                minHWidth = min(h1[2], h2[2])
                minVHeight = min(v1[3], v2[3])
                cornerRadius = min(minHWidth, minVHeight) * 0.25
                
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

# ---------------- Modified: Draw rectangles & remove actual line areas ----------------
def draw_rectangles(image, rectangles, line_areas_to_remove, bg_color=(255, 255, 255)):
    image_rect = image.copy()
    
    # Remove the actual line areas that form the rectangles
    for line_area in line_areas_to_remove:
        x1, y1, x2, y2 = line_area
        # Fill the actual line area with background color
        image_rect[y1:y2, x1:x2] = bg_color
        
    return image_rect

def classify_relationship(contours, hierarchy, idx):
    """
    Classify a contour (line + attached symbol) into UML relationship type.
    Returns classification and points (start/end for directed, endpoints for plain associations).
    """
    
    global epsilon_ratio
    cnt = contours[idx]
    child_idx = hierarchy[0][idx][2]
    has_five_seq, convexHull = None, None
    epsilon_ratio = 0.04
    approx = None
    while True:
        # Get approximate points for the contour
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        vertices = len(approx)
        
        # Find two farthest points (endpoints for all line types)
        farthest_points = find_farthest_points(approx)
        if farthest_points is None:
            return "Association (plain line)", None, None, None, None, approx
        
        point1, point2, max_distance = farthest_points
        
        # If no 5-point sequence, check if it has a child symbol
        if child_idx != -1:
            child_cnt = contours[child_idx]
            child_epsilon = 0.08 * cv2.arcLength(child_cnt, True)
            child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
            child_vertices = len(child_approx)
            # Classify symbol type
            if child_vertices == 3:
                relationship_type = "inheritance"
            elif child_vertices == 4:
                relationship_type = "aggregation"
            else:
                relationship_type = "Other unfilled shape"
            
            # Determine start/end: which farthest point is closer to child symbol
            start_point, end_point = determine_start_end_with_child(point1, point2, child_approx)
            
            return relationship_type, start_point, end_point, point1, point2, approx
        
        
        # First, check for 5-point sequence (this takes priority)
        has_five_seq, convexHull = find_five_point_sequence(approx)
        
        if has_five_seq:
            break
        elif epsilon_ratio <= 0.005:
            break 
        
        epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx
    
    if has_five_seq:
        if convexHull and convexHull == 3:
            relationship_type = "directed association"
        else:
            relationship_type = "composition"
        # Determine start/end for 5-point sequence relationships
        start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
        return relationship_type, start_point, end_point, point1, point2, approx
    else:
        # If no 5-point sequence and no child symbol, it's a plain association
        relationship_type = "association"
        # Return endpoints but no start/end direction
        return relationship_type, None, None, point1, point2, approx

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

def determine_start_end_with_child(point1, point2, child_approx):
    """Determine start/end for contours with child symbols."""
    child_points = np.squeeze(child_approx)
    
    if len(child_points) == 0:
        return point1, point2  # Fallback
    
    # Calculate distances from each point to the child symbol
    dist1 = min([np.linalg.norm(np.array(point1) - child_point) for child_point in child_points])
    dist2 = min([np.linalg.norm(np.array(point2) - child_point) for child_point in child_points])
    
    # The point closer to the child symbol is the end (arrowhead/diamond side)
    if dist1 < dist2:
        return point2, point1  # point1 is end (closer to symbol)
    else:
        return point1, point2  # point2 is end (closer to symbol)

def determine_start_end_five_sequence(point1, point2, approx):
    """Determine start/end for 5-point sequence (filled diamond)."""
    points = np.squeeze(approx)
    
    # Find the 5-point sequence
    five_seq_points, _ = find_five_point_sequence(points)
    
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

def find_five_point_sequence(points, min_tolerance=5, tolerance_percent=0.25):
    """Find and return the 5-point sequence if it exists."""
    n = len(points)
    if n < 5:
        return None, None
    
    for start in range(n+(n-1)):
        distances = []
        seq_points = []
        
        for i in range(5):
            p1 = points[(start + i) % n]
            seq_points.append(p1)
        
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
            if hull_count >= 4:
                return seq_points, hull_count
            elif hull_count == 3:
                return seq_points, hull_count
            
    return None, None

def detect_contours(image, rectangles, bg_color=0, min_area=100):
    detectedContours = image.copy()
    detectText = image.copy()
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    ) 

    contour_img = cv2.cvtColor(detectedContours, cv2.COLOR_GRAY2BGR)
    contours_info = []
    count = 0
    classHorizontalLineSeparators = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        if w > 25 or h > 25:  # filter small noise
            
            inTheRect = False
            for rect_index, rect in enumerate(rectangles):
                rx1, ry1, rx2, ry2 = rect
                # Check if contour bbox is within rectangle bbox
                if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                    # Fill rectangle area with background color to remove it
                    classHorizontalLineSeparators.append((x, y, w, h, rect_index)) # store horizontal line info and the rect's id
                    cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
                    inTheRect = True
                    break  # No need to check other rectangles
                
            if inTheRect: continue  # Skip this contour entirely
            
            parent = hierarchy[0][i][3]
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
            
            if parent != -1:
                continue  # skip child contours
                
            count += 1
            classification, start_point, end_point, endpoint1, endpoint2, approx = classify_relationship(
                contours, hierarchy, i
            )

            # Draw approx points
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Save info for merging - ONLY CHANGE: added has_arrow_head flag
         
            contours_info.append({
                "type": classification,
                "start": start_point,
                "end": end_point,
                "p1": endpoint1,
                "p2": endpoint2,
                "approx": approx,
                "has_arrow_head": classification in ["aggregation", "composition", "inheritance", "dependency", "directed association"]  # ONLY ADDITION
            })
    
    return [contours_info, classHorizontalLineSeparators], hierarchy, detectedContours, contour_img, detectText

def getTexts(rect, texts, horizontalLineSeperators):
    
    attributes = []
    methods = []
    class_name = ""
    
    # texts within the rectangle 
    attributeTexts = []
    methodTexts = []
    classNameTexts = []
    horizontalLineSeperators.sort(key=lambda line: line[1])
    upperLineY = horizontalLineSeperators[0]
    lowerLineY = horizontalLineSeperators[1] if len(horizontalLineSeperators) > 1 else None
    
    text_to_remove = []
    for text in texts:
        tx, ty, tw, th, content = text
        rx1, ry1, rx2, ry2 = rect
        
        if tx >= rx1 and (tx + tw) <= rx2 and ty >= ry1 and (ty + th) <= ry2:
            # check if the text is above the first horizontal line separator
            if ty < upperLineY[1]:
                classNameTexts.append(text)
            # check if the text is between the two horizontal line separators
            elif lowerLineY is not None and ty >= upperLineY[1] and (ty + th) <= lowerLineY[1]:
                attributeTexts.append(text)
            # check if the text is below the second horizontal line separator
            elif lowerLineY is not None and ty >= lowerLineY[1]:
                methodTexts.append(text)
            elif lowerLineY is None and ty >= upperLineY[1]:
                methodTexts.append(text)
            
            text_to_remove.append(text)  # mark this text for removal
    
    # Remove processed texts using slice assignment to modify the original list
    texts[:] = [t for t in texts if t not in text_to_remove]
    
    # Helper function to group texts by lines and form text strings
    def group_texts_by_line(textType, text_list):
        if not text_list or len(text_list) < 1:
            return []
        
        y_tolerance = 5
        texts_sorted = sorted(text_list, key=lambda t: (t[1], t[0]))
        lines = []
        current_line = [texts_sorted[0]]
        
        for j in range(1, len(texts_sorted)):
            current_text = texts_sorted[j]
            prev_text = texts_sorted[j-1]
            
            # Calculate Y centers
            current_center_y = current_text[1] + current_text[3] / 2
            prev_center_y = prev_text[1] + prev_text[3] / 2
                
            # Check if Y difference is within threshold (same line)
            if abs(current_center_y - prev_center_y) <= y_tolerance:
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
        
        # Build text strings for each line
        result = []
        for line in lines:
            line_text = ""
            access_modifier = "~"
            for i, text in enumerate(line):
                _, _, _, _, txt = text
                
                if textType == "attribute" or textType == "method":
                    # get the access modifier
                    if(i == 0): 
                        if txt and txt[0] in "+-#~":
                            access_modifier = txt[0]
                            
                        elif len(txt) > 1: 
                            if txt and txt[0] in "+-#~":
                                access_modifier = txt[0]
                                line_text += " " + txt[1:]
                            else:
                                line_text += " " + txt
                    else:
                        line_text = line_text + " " + txt

                else:
                    line_text = line_text + " " + txt
            
            if textType == "name":
                result.append(line_text.strip())
            else:
                result.append([access_modifier, line_text.strip()])
        return result
    
    # Process class name (special case - sin gle string)
    if classNameTexts and len(classNameTexts) > 0:
        class_name_lines = group_texts_by_line("name", classNameTexts)
        class_name = " ".join(class_name_lines)  # Join all lines for class name
    else:
        class_name = ""
    
    # Process attributes and methods using the same helper function
    attributes = group_texts_by_line("attribute", attributeTexts)
    methods = group_texts_by_line("method", methodTexts)
            
    return attributes, methods, class_name, texts

# to merge nearby text and form sentence
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

    final_merged = merged_horizontal
    
    return final_merged

def to_uniform_point(point):
    """Convert any point format to (np.int32(x), np.int32(y))"""
    if point is None:
        return (np.int32(0), np.int32(0))
    if isinstance(point, tuple) and len(point) == 2:
        if isinstance(point[0], np.int32) and isinstance(point[1], np.int32):
            return point  # Already in correct format
        else:
            return (np.int32(point[0]), np.int32(point[1]))
    elif hasattr(point, '__iter__') and len(point) == 2:
        return (np.int32(point[0]), np.int32(point[1]))
    else:
        return (np.int32(0), np.int32(0))

def mergeTransition(transitions, texts, margin = 15):
    
    for transition in transitions:
        # Get bounding box of transition
        tx1, ty1, tw, th = cv2.boundingRect(transition["approx"])
        tx2, ty2 = tx1 + tw, ty1 + th
        
        # Expand transition bbox with margin
        trans_bbox_expanded = (tx1 - margin, ty1 - margin, tx2 + margin, ty2 + margin)
        ex1, ey1, ex2, ey2 = trans_bbox_expanded

        transition["labels"] = []
        
        for text in texts:
            bx, by, bw, bh, text_content = text
            text_bbox = (bx, by, bx + bw, by + bh)
            bx1, by1, bx2, by2 = text_bbox 
            # Check if text bbox intersects with expanded transition bbox
            if (bx1 < ex2 and bx2 > ex1 and by1 < ey2 and by2 > ey1):                
                transition["labels"].append(text)
    
    # merge and remove taht has same label's property
    removeTransitions = []
    mergeTransitions = []
    
    for i in range(len(transitions)):
        if transitions[i].get("labels") is None or len(transitions[i]["labels"]) == 0:
            continue
        
        for j in range(i+1, len(transitions)):
            if transitions[j].get("labels") is None or len(transitions[j]["labels"]) == 0:
                continue
            
            for i_label in transitions[i]["labels"]:
                for j_label in transitions[j]["labels"]:
                    if(i_label == j_label):            
                        # should not merge if both have start and end points
                        if(transitions[i]["has_arrow_head"] and transitions[j]["has_arrow_head"]):
                            continue
                        
                        relationshipType = "association"
                        arrowhead = None
                        startPoint = None
                        
                        iPoint1, iPoint2 = None, None
                        if(transitions[i].get("start") is None):
                            iPoint1, iPoint2 = transitions[i]["p1"], transitions[i]["p2"]
                        else:
                            iPoint1, iPoint2 = transitions[i].get("start"), transitions[i].get("end")
                            arrowhead, startPoint = iPoint2, iPoint1
                        
                        jPoint1, jPoint2 = None, None
                        if(transitions[j].get("start") is None):
                            jPoint1, jPoint2 = transitions[j]["p1"], transitions[j]["p2"]
                        else:
                            jPoint1, jPoint2 = transitions[j].get("start"), transitions[j].get("end")
                            arrowhead, startPoint = jPoint2, jPoint1
                        
                        # Convert all points to uniform format using the helper function
                        iPoint1 = to_uniform_point(iPoint1)
                        iPoint2 = to_uniform_point(iPoint2)
                        jPoint1 = to_uniform_point(jPoint1)
                        jPoint2 = to_uniform_point(jPoint2)
                        
                        if arrowhead is not None:
                            arrowhead = to_uniform_point(arrowhead)
                        if startPoint is not None:
                            startPoint = to_uniform_point(startPoint)
                        
                        # Convert to numpy arrays for calculations
                        iPoint1_arr = np.array(iPoint1)
                        iPoint2_arr = np.array(iPoint2)
                        jPoint1_arr = np.array(jPoint1)
                        jPoint2_arr = np.array(jPoint2)
                        arrowhead_arr = np.array(arrowhead) if arrowhead is not None else None
                        startPoint_arr = np.array(startPoint) if startPoint is not None else None
                        
                        p1_arr, p2_arr = None, None
                        
                        if arrowhead_arr is not None:
                            if(transitions[i].get("has_arrow_head") is not None):
                                relationshipType = transitions[i].get("type")
                            elif(transitions[j].get("has_arrow_head") is not None):
                                relationshipType = transitions[j].get("type")

                            if(transitions[i].get("end") is not None):
                                arrowhead_arr = np.array(transitions[i]["end"])
                                # For plain line j, find the point farthest from the arrowhead
                                arrowhead_to_P1 = np.linalg.norm(arrowhead_arr - jPoint1_arr)
                                arrowhead_to_P2 = np.linalg.norm(arrowhead_arr - jPoint2_arr)
                                
                                # Choose the point that is farthest from the arrowhead
                                if arrowhead_to_P1 > arrowhead_to_P2:
                                    startPoint_arr = jPoint1_arr  # Use the actual point, not the distance
                                else:
                                    startPoint_arr = jPoint2_arr  # Use the actual point, not the distance
                                
                                p1_arr = np.array(startPoint_arr)
                                p2_arr = np.array(arrowhead_arr)
                                
                            else:
                                # j has arrowhead, i is plain line
                                arrowhead_arr = np.array(transitions[j]["end"])
                                # For plain line i, find the point farthest from the arrowhead
                                arrowhead_to_P1 = np.linalg.norm(arrowhead_arr - iPoint1_arr)
                                arrowhead_to_P2 = np.linalg.norm(arrowhead_arr - iPoint2_arr)
                                
                                startPoint = max(arrowhead_to_P1, arrowhead_to_P2)
                                
                                p1_arr = np.array(startPoint)
                                p2_arr = np.array(arrowhead_arr)
                        else:
                            point1, point2, _ = find_farthest_points(np.array([iPoint1_arr, iPoint2_arr, jPoint1_arr, jPoint2_arr]))
                            p1_arr = point1
                            p2_arr = point2
                        
                        removeTransitions.append(transitions[i])
                        removeTransitions.append(transitions[j])
                        
                        # Convert final points back to uniform np.int32 format
                        startPoint_final = to_uniform_point(startPoint_arr) if startPoint_arr is not None else None
                        arrowhead_final = to_uniform_point(arrowhead_arr) if arrowhead_arr is not None else None
                        p1_final = to_uniform_point(p1_arr) if p1_arr is not None else to_uniform_point(iPoint1_arr)
                        p2_final = to_uniform_point(p2_arr) if p2_arr is not None else to_uniform_point(iPoint2_arr)
                        
                        # Create merged transition with uniform np.int32 format
                        mergeTransitions.append({
                            "type": relationshipType,
                            "start": startPoint_final,
                            "end": arrowhead_final,
                            "p1": p1_final,
                            "p2": p2_final,
                            "approx": [np.array([p1_final[0], p1_final[1]]), 
                                      np.array([p2_final[0], p2_final[1]])],
                            "labels": transitions[i]["labels"]
                        })
                        
                        break  # break inner loop after merging
    
    for transition in removeTransitions:
        if transition in transitions:
            transitions.remove(transition)
    
    for transition in mergeTransitions:
        transitions.append(transition)
    
    return transitions

def mergeBrokenLines(contours):
    threshold = 5  # number of pixels to consider as one contour in line
    
    while True:
        if len(contours) < 2:
            break
        
        hasMerged = False
        merged_indices = set()  # Keep track of merged contours to avoid processing them again
        
        # Create a copy of contours list for safe iteration
        contours_copy = contours.copy()
        
        for i, cnt1 in enumerate(contours_copy):
            if i in merged_indices:
                continue
                
            p1_end = np.array(cnt1['p2'])
            p1_start = np.array(cnt1['p1'])
            has_arrow_head_1 = cnt1.get('has_arrow_head', False)
            
            for j, cnt2 in enumerate(contours_copy):
                if i == j or j in merged_indices:
                    continue
                
                p2_end = np.array(cnt2['p2'])
                p2_start = np.array(cnt2['p1'])
                has_arrow_head_2 = cnt2.get('has_arrow_head', False)
                
                # Skip merging if two contours have arrow heads
                if has_arrow_head_1 and has_arrow_head_2:
                    continue
                
                # Check all combinations of endpoints
                distances = [
                    (np.linalg.norm(p1_end - p2_start), [p1_end, p2_start], [p1_start, p2_end]),
                    (np.linalg.norm(p1_start - p2_end), [p1_start, p2_end], [p1_end, p2_start]),
                    (np.linalg.norm(p1_start - p2_start), [p1_start, p2_start], [p1_end, p2_end]),
                    (np.linalg.norm(p1_end - p2_end), [p1_end, p2_end], [p1_start, p2_start])
                ]
                
                # Find the minimum distance connection
                min_dist = float('inf')
                best_points = None
                best_newEndpoints = None
                
                for dist, points, newEndpoints in distances:
                    if dist < min_dist and dist < threshold:
                        min_dist = dist
                        best_points = points
                        best_newEndpoints = newEndpoints
                
                if best_points is not None:  # Found a valid merge
                    relationshipType = "association"
                    
                    if has_arrow_head_1:
                        # Convert to tuples for comparison
                        end_point_tuple = tuple(cnt1['end']) if isinstance(cnt1['end'], np.ndarray) else cnt1['end']
                        point0_tuple = tuple(best_points[0]) if isinstance(best_points[0], np.ndarray) else best_points[0]
                        point1_tuple = tuple(best_points[1]) if isinstance(best_points[1], np.ndarray) else best_points[1]
                        
                        if point0_tuple == end_point_tuple or point1_tuple == end_point_tuple:
                            continue  # skip if arrow head is at the merging point
                        else:
                            newEndpoint = cnt1['end']
                            newStartPoint = best_newEndpoints[0] if not np.array_equal(best_newEndpoints[0], newEndpoint) else best_newEndpoints[1]
                            relationshipType = cnt1['type']
                            
                    elif has_arrow_head_2:
                        # Convert to tuples for comparison
                        end_point_tuple = tuple(cnt2['end']) if isinstance(cnt2['end'], np.ndarray) else cnt2['end']
                        point0_tuple = tuple(best_points[0]) if isinstance(best_points[0], np.ndarray) else best_points[0]
                        point1_tuple = tuple(best_points[1]) if isinstance(best_points[1], np.ndarray) else best_points[1]
                        
                        if point0_tuple == end_point_tuple or point1_tuple == end_point_tuple:
                            continue  # skip if arrow head is at the merging point
                        else:
                            newEndpoint = cnt2['end']
                            newStartPoint = best_newEndpoints[0] if not np.array_equal(best_newEndpoints[0], newEndpoint) else best_newEndpoints[1]
                            relationshipType = cnt2['type']
                    else:
                        newEndpoint = best_newEndpoints[0]
                        newStartPoint = best_newEndpoints[1]
                    
                    # Convert to uniform np.int32 format using the helper function
                    p1 = to_uniform_point(newStartPoint)
                    p2 = to_uniform_point(newEndpoint)
                    
                    # Create new merged contour with uniform formatting
                    new_contour = {
                        'type': relationshipType,
                        'start': to_uniform_point(newStartPoint),
                        'end': to_uniform_point(newEndpoint),
                        'p1': p1,
                        'p2': p2,
                        'approx': np.array([p1, p2], dtype=np.int32), 
                        'has_arrow_head': has_arrow_head_1 or has_arrow_head_2
                    }
                    
                    # Mark these contours for removal
                    merged_indices.add(i)
                    merged_indices.add(j)
                    hasMerged = True
                    
                    # Add the new merged contour
                    contours.append(new_contour)
                    break  # Break inner loop after finding a merge for cnt1
            
            if hasMerged:
                # Rebuild the contours list by removing merged contours
                new_contours = [cnt for idx, cnt in enumerate(contours_copy) if idx not in merged_indices]
                # Add the newly created merged contours
                new_contours.extend([cnt for cnt in contours if cnt not in contours_copy])
                contours[:] = new_contours  # Update the original list
                break  # Break outer loop and restart while loop
        
        if not hasMerged:
            break
                        
    return contours


def getConnectedNode(classNodes, point, relationshipID, pos, margin = 10):
    
    # for all arrow line
    for node in classNodes:
        x, y, w, h = node["bbox"]
        px, py = int(point[0]), int(point[1])
        
        def getSide(point, nodeBbox):
            pointX, pointY = point[0], point[1]
            x, y, w, h = nodeBbox

            positions = [0.20, 0.50, 0.80, 1.00]

            handle_positions = {
                **{f"top-{i}": (x + w * p, y) for i, p in enumerate(positions)},
                **{f"bottom-{i}": (x + w * p, y + h) for i, p in enumerate(positions)},
                **{f"left-{i}": (x, y + h * p) for i, p in enumerate(positions)},
                **{f"right-{i}": (x + w, y + h * p) for i, p in enumerate(positions)},
            }
            closestSide = min(
                handle_positions,
                key=lambda key: math.hypot(px - handle_positions[key][0], py - handle_positions[key][1])
            )
            
            # side = min(distances, key=distances.get)
            return closestSide
             
        if((x-margin <= px) and (x+w+margin >= px) and (y-margin <= py) and (y+h+margin > py)):
            
            # inside the action
            node["hasConnection"] = True
            
            if pos not in node:
                node[pos] = []
                
            node[pos].append({
                "transitionID": relationshipID
            })
            side = getSide(point, node["bbox"])
                
            return node["id"], side

    return None


def restructureData(contours, rectangles, extractedTexts, horizontalLineSeparators):
    # merge broken lines
    contours = mergeBrokenLines(contours)    

    classNodes = []
    relationships = []
    
    texts = extractedTexts
    
    # Process rectangles as class nodes
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        
        # finding the line separators that belongs to the current rectangle/class
        lineSeparator = []
        for horizontalLine in horizontalLineSeparators:
            if(horizontalLine[4] == i):   #check if the horizontal line belongs to the current rectangle
                lineSeparator.append(horizontalLine)
                 
        attributes, methods, class_name, texts = getTexts(rect, texts, lineSeparator)
        
        classNodes.append({
            "id": str(len(classNodes) + 1),
            "type": "class",
            "label": class_name,
            "bbox": (x1, y1, x2, y2),
            "methods": methods,
            "attributes": attributes,
            "to": [],
            "from": []
        })
    
    # merged close text outside from the class, it could be the multiplicity
    text_merged = merge_nearby_texts(texts)

    # process relationship
    for cnt in contours:
        relationships.append(cnt)
    
    relationships = mergeTransition(relationships, text_merged)
        
    for i, relationship in enumerate(relationships): 
        relationship["id"] = "line" + str(i + 1)
    
    for i, relationship in enumerate(relationships):
        relationship["middleLabel"], relationship["startLabel"], relationship["endLabel"] = None, None, None
        start = relationship.get("start") if relationship.get("start") is not None else  relationship.get("p1")
        end = relationship.get("end") if relationship.get("end") is not None else  relationship.get("p2")
        
        startingNodeID = getConnectedNode(classNodes, start, relationship.get("id"), "from")
        destinationNodeID = getConnectedNode(classNodes, end, relationship.get("id"), "to")
        
        if startingNodeID and destinationNodeID:
            relationship["startNodeID"] = startingNodeID[0]
            relationship["endNodeID"] = destinationNodeID[0]
            relationship["sourceHandle"] = startingNodeID[1] 
            relationship["targetHandle"] = destinationNodeID[1] 

        def get_label_position(label, p1, p2, percentage_threshold=0.3):
            x, y, w, h, text = label
            label_center = np.array([x + w/2, y + h/2])
            
            dist_to_p1 = np.linalg.norm(label_center - p1)
            dist_to_p2 = np.linalg.norm(label_center - p2)
            
            # Convert p1 and p2 to numpy arrays
            p1_arr = np.array(p1)
            p2_arr = np.array(p2)
            
            # Calculate the total line length
            line_length = dist_to_p1 + dist_to_p2
             
            # Calculate the difference as a percentage of the line length
            min_distance = min(dist_to_p1, dist_to_p2)
            diff_percentage = min_distance / line_length
            
            # Check if the difference is small relative to the line length
            if diff_percentage >= percentage_threshold:
                return "middle", min(dist_to_p1, dist_to_p2)
            elif dist_to_p1 < dist_to_p2:
                return "p1", dist_to_p1
            else:
                return "p2", dist_to_p2

        # Usage:
        for label in relationship["labels"]:
            position, distance = get_label_position(label, relationship["p1"], relationship["p2"])
            if position == "p1":
                if relationship["p1"] is relationship["start"]:
                    relationship["startLabel"] = label[4]
                else: 
                    relationship["endLabel"] = label[4]
            elif position == "p2":
                if relationship["p2"] is relationship["start"]:
                    relationship["startLabel"] = label[4]
                else: 
                    relationship["endLabel"] = label[4]
            else:
                relationship["middleLabel"] = label[4]


    def getAccessibilityModifiers(attributesOrMethods):
        result = []
        for item in attributesOrMethods:
          
            access_modifier, text = item
            result.append({
                "access": access_modifier,
                "value": text
            })
        
        return result
    
    structuredData = {
        "nodes": [],
        "edges": []
    }
    
    for node in classNodes:
        
        x, y, w, h = node.get("bbox", (0,0,0,0))
        w = w - x
        h = h - y
        classNode = {
            "id": node.get("id"),
            "type": "ClassNode",
            "position": {"x": x, "y": y},
            "data": { 
                "attributes": getAccessibilityModifiers(node.get("attributes", [])),
                "className": node.get("label", " "),
                "methods": getAccessibilityModifiers(node.get("methods", [])),
            },
            "style": {"width": w, "height": h},
            "measured": {"width": w, "height": h},
        }
        
        print(classNode["data"]["className"], x, y, w, h)
        structuredData["nodes"].append(classNode)
        
    def getSymbols(relationshipType):
        startSymbol = "none"
        endSymbol = "none"
        lineStyle = "line"
        
        if relationshipType == "association":
            startSymbol = "none"
            endSymbol = "none"
            lineStyle = "line"
        elif relationshipType == "directed association":
            startSymbol = "none"
            endSymbol = "open arrow"
            lineStyle = "line"
        elif relationshipType == "inheritance":
            startSymbol = "none"
            endSymbol = "closed arrow"
            lineStyle = "line"
        elif relationshipType == "dependency":
            startSymbol = "none"
            endSymbol = "open arrow"
            lineStyle = "dashLine"
        elif relationshipType == "aggregation":
            startSymbol = "none"
            endSymbol = "open diamond"
            lineStyle = "line"
        elif relationshipType == "composition":
            startSymbol = "none"
            endSymbol = "filled diamond"
            lineStyle = "line"
        
        return startSymbol, endSymbol, lineStyle
    
    
    for relationship in relationships:
        startSymbol, endSymbol, lineStyle = getSymbols(relationship.get("type"))
        
        edge = {
            "id": relationship.get("id"),
            "type": "edge",
            "zIndex": 1000,
            "style": {"zIndex": 100},
            "source": relationship.get("startNodeID"),
            "sourceHandle": relationship.get("sourceHandle"),
            "target": relationship.get("endNodeID"),
            "targetHandle": relationship.get("targetHandle"),
            "data": {
                "diagramType": "class",
                "sourceHandle": relationship.get("sourceHandle"),
                "targetHandle": relationship.get("targetHandle"),
                "startLabel": relationship.get("startLabel") if relationship.get("startLabel") is not None else " ",
                "middleLabel": relationship.get("middleLabel") if relationship.get("middleLabel") is not None else " ",
                "endLabel": relationship.get("endLabel") if relationship.get("endLabel") is not None else " ",
                "startSymbol": startSymbol,
                "endSymbol": endSymbol,
                "stepLine": True,
                "lineStyle": lineStyle,
                # "sourceX": int(relationship.get("start")[0]) if relationship["start"] is not None else None,
                # "sourceY": int(relationship.get("start")[1]) if relationship["start"] is not None else None,
                # "targetX": int(relationship.get("end")[0]) if relationship["end"] is not None else None,
                # "targetY": int(relationship.get("end")[1]) if relationship["end"] is not None else None,
            },
        }
        
        structuredData["edges"].append(edge)
    
    return structuredData


import psutil, os

# ---------------- Main workflow ----------------
def process_class_diagram(file):
    
    start_time = time.time()
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024
    
    
    print("Processing class diagram...")
    image, gray = load_image_from_file(file)
    thresh = threshold_image(gray)
    vertical_lines, horizontal_lines = extract_lines(thresh)
    print("vertical processing")
    # get contours to all detected vertical and horizontal lines
    vertical_segments = get_segments(vertical_lines)
    horizontal_segments = get_segments(horizontal_lines)
    print("line pairs")
    # pairs those have almost identical vertical/horizontal lines, and get those lines that has no pairs
    vertical_pairs = find_vertical_pairs(vertical_segments)
    horizontal_pairs = find_horizontal_pairs(horizontal_segments)
    print(" finding rect")
    # get possible rectangles based on two pairs of vertical and horizontal lines
    rectangles, lines_to_remove = detect_rectangles(vertical_pairs, horizontal_pairs, vertical_segments, horizontal_segments)
    # remove/draw rectangles in the image
    image_rect = draw_rectangles(image, rectangles, lines_to_remove)
    
    # ==================== Preprocess again the image that after removing teh rect =============================== 
    noRectImg = image_rect.copy()      #copy image after removing rectangles
    threshImg = threshold_image(cv2.cvtColor(noRectImg, cv2.COLOR_BGR2GRAY))    #threshold again
    print("Rect:", len(rectangles))
    extractedContours, hierarchy, detectedContours, drawContour, textImg = detect_contours(threshImg, rectangles)
    
    contours, classHorizontalSeparators = extractedContours[0], extractedContours[1]
    
    # --- 3. OCR + bounding box info ---
    textData = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    
    # --- 5. Extract text data ---
    texts = []
    
    for i in range (len(textData['text'])):
        text = textData['text'][i].strip()
        if text != "":  # ignore empty OCR outputs
            
            x = int(textData['left'][i])
            y = int(textData['top'][i])
            w = int(textData['width'][i])
            h = int(textData['height'][i])
            texts.append((x, y, w, h, text))
    
    structuredData = restructureData(contours, rectangles, texts, classHorizontalSeparators)
   
    end_time = time.time()
    
    executionTime = end_time - start_time
    after = process.memory_info().rss / 1024 / 1024
    memory_used = abs(after - before)
    
    structuredData["executionTime"] = f"{executionTime} second/s"
    structuredData["memory_usage"] = f"{memory_used}" 
    
    
    return structuredData