import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json
import time

epsilon_ratio = 0.04
image_height = 0
image_width = 0
actors = []
removeVerticalLines = []
removeHorizontalLines = []

def load_image(file):
    """Load image from file object"""
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    global image_height, image_width
    image_height, image_width = gray.shape[:2]

    return image, gray

def threshold_image(gray, thresh_val=128):
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV) 
    return thresh

def extract_lines(thresh, vert_len=15, horiz_len=15):
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    return vertical_lines, horizontal_lines

def get_segments(lines):
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [cv2.boundingRect(c) for c in contours]
    return segments

def find_vertical_pairs(segments, image, y_tol=5, h_tol=5, bg_color=(255, 255, 255)):
    new_image = image.copy()
    allowableHeights = image_height * 0.75
    global removeVerticalLines
    
    for segment in segments:
        if segment[3] > allowableHeights:
            x, y, w, h = segment
            new_image[y-2:y+h+2, x-2:x+w+2] = bg_color
            removeVerticalLines.append(segment)
        
    segments = [s for s in segments if s not in removeVerticalLines]
    pairs = []
    
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(y1 - y2) <= y_tol and abs(h1 - h2) <= h_tol:
                pairs.append((segments[i], segments[j]))

    return pairs, new_image

def find_horizontal_pairs(segments, image, x_tol=5, w_tol=5, bg_color=(255, 255, 255)):
    new_image = image.copy()
    allowableWidth = image_width * 0.75
    global removeHorizontalLines
    
    for segment in segments:
        x, y, w, h = segment
        if w > allowableWidth:
            removeHorizontalLines.append(segment)
            new_image[y-2:y+h+2, x-2:x+w+2] = bg_color
        else:    
            if h < 10:
                new_image[y:y+h, x:x+w] = (0, 0, 0)
            
    segments = [s for s in segments if s not in removeHorizontalLines]
    pairs = []
    
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(x1 - x2) <= x_tol and abs(w1 - w2) <= w_tol:
                pairs.append((segments[i], segments[j])) 
    
    return pairs, new_image

def detect_rectangles(vertical_pairs, horizontal_pairs, vertical_lines, horizontal_lines):
    rectangles = []
    line_areas_to_remove = []
    
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
                
                if (maxWidth - h1Width) < 40 and (maxHeight - v1Height) < 40:
                    rectangles.append((v_left_x, h_top_y, v_right_x, h_bottom_y))
                    line_areas_to_remove.append((h_left_x, h_top_y, h_right_x, h_top_y + max(h1[3], h2[3])))
                    line_areas_to_remove.append((h_left_x, h_bottom_y - max(h1[3], h2[3]), h_right_x, h_bottom_y))
                    line_areas_to_remove.append((v_left_x, v_top_y, v_left_x + max(v1[2], v2[2]), v_bottom_y))
                    line_areas_to_remove.append((v_right_x - max(v1[2], v2[2]), v_top_y, v_right_x, v_bottom_y))
                    
    return rectangles, line_areas_to_remove

def remove_rectangles(image, line_areas_to_remove, vLines, hLines, bg_color=(255, 255, 255)):
    image_rect = image.copy()
    
    for line_area in line_areas_to_remove:
        x1, y1, x2, y2 = line_area
        image_rect[y1:y2, x1:x2] = bg_color

    for vLine in vLines:
        vx, vy, vw, vh = vLine
        if vw < 10: continue
        
        for hLine in hLines:
            hx, hy, hw, hh = hLine
            if hh < 10: continue

            if (vx < hx + hw and vx + vw > hx and
                vy < hy + hh and vy + vh > hy):
                diameter = min(vh, hw) 
                radius = diameter // 2
                
                if vh < hw:
                    image_rect[vy + (vh // 2) - 5: vy + (vh // 2) + 5, vx + (vw//2) + radius] = bg_color
                    image_rect[vy + (vh // 2) - 5: vy + (vh // 2) + 5, vx + (vw//2) - radius] = bg_color
                else:
                    image_rect[hy + (hh // 2) - radius, hx + ( hw//2 ) - 5: hx + (hw//2) + 5] = bg_color
                    image_rect[hy + (hh // 2) + radius, hx + ( hw//2 ) - 5: hx + (hw//2) + 5] = bg_color

    return image_rect

def find_farthest_points(approx_points):
    points = np.squeeze(approx_points)
    
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
    points = np.squeeze(approx)
    five_seq_points, _ = find_five_point_sequence(points, point1, point2)
    
    if five_seq_points is None:
        return point1, point2

    point1_in_seq = any(np.array_equal(np.array(point1), seq_point) for seq_point in five_seq_points)
    point2_in_seq = any(np.array_equal(np.array(point2), seq_point) for seq_point in five_seq_points)
    
    if point1_in_seq and not point2_in_seq:
        return point2, point1
    elif point2_in_seq and not point1_in_seq:
        return point1, point2
    else:
        return None, None

def find_five_point_sequence(points, point1, point2, min_tolerance=5, tolerance_percent=0.25):
    n = len(points)
    if n < 5:
        return None, None

    for start in range(n):
        distances = []
        seq_points = []
        
        for i in range(5):
            p1 = points[(start + i) % n]
            seq_points.append(p1)
        
        point1_in_seq = any(np.array_equal(np.array(point1), seq_point) for seq_point in seq_points)
        point2_in_seq = any(np.array_equal(np.array(point2), seq_point) for seq_point in seq_points)
        
        if point1_in_seq and point2_in_seq:
            continue
        
        for i in range(4):
            p1 = seq_points[i]
            p2 = seq_points[i + 1]
            dist = np.linalg.norm(p1 - p2)
            distances.append(dist)
        
        max_dist = max(distances)
        min_dist = min(distances)
        
        original_max = 30
        original_min = 18
        scale_factor = max_dist / original_max
        scaled_min = original_min * scale_factor
        
        dynamic_tolerance = max_dist - scaled_min
        
        if dynamic_tolerance < 5:
            return None, None
        
        if max(distances) - min(distances) <= dynamic_tolerance+5:
            seq_np = np.array(seq_points, dtype=np.int32)
            hull = cv2.convexHull(seq_np, returnPoints=True)
            hull_count = len(hull)
      
            if hull_count == 3:
                return seq_points, hull_count
       
    return None, None

def classify_based_on_child(contours, hierarchy, idx):
    cnt = contours[idx]
    epsilon = 0.08 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
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
    
    if len(children) == 4:
        return "destruction node", None, None, approx
    elif len(children) == 1 and len(grand_children) > 0:
        return "end node", None, None, approx
    
    child_idx = hierarchy[0][idx][2]
    while child_idx != -1:
        child_cnt = contours[child_idx]
        x, y, w, h = cv2.boundingRect(child_cnt)
        child_epsilon = 0.08 * cv2.arcLength(child_cnt, True)
        child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
        
        if len(child_approx) == 4 and w/h > .9:
            return "diamond node", None, None, approx
        
        child_idx = hierarchy[0][child_idx][0]
    
    return "line", None, None, approx

def classify_contour(contours, hierarchy, idx):
    global epsilon_ratio
    
    child_idx = hierarchy[0][idx][2]
    
    if child_idx != -1:
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
    
    perimeter = cv2.arcLength(cnt, True)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0

    if bbox_area > 0 and (area / bbox_area) >= 0.75 and circularity >= 0.7:
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)   
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return "start node", None, None, approx
    
    while True:
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)   
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        farthest_points = find_farthest_points(approx)
        if farthest_points is None:
            return "Unknown", None, None, approx
        
        point1, point2, max_distance = farthest_points 
        has_five_seq, convexHull = find_five_point_sequence(approx, point1, point2)

        if has_five_seq:
            break
        elif epsilon_ratio <= 0.005:
            break

        epsilon_ratio = epsilon_ratio / 2

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
    
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    ) 

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        child_idx = hierarchy[0][i][2]
        while child_idx != -1:
            child_cnt = contours[child_idx]
            cx, cy, cw, ch = cv2.boundingRect(child_cnt)

            if cw > 25 and ch > 25:
                epsilon = 0.08 * cv2.arcLength(child_cnt, True)
                approx = cv2.approxPolyDP(child_cnt, epsilon, True)

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

                if len(approx) == 4:
                    middle_y = cy + ch // 2
                    middle_x = cx + cw // 2

                    image[middle_y-3:middle_y+3, cx-4] = (255, 255, 255)
                    image[middle_y-3:middle_y+3, cx+cw+4] = (255, 255, 255)
                    image[cy-4, middle_x-3:middle_x+3] = (255, 255, 255)
                    image[cy+ch+4, middle_x-3:middle_x+3] = (255, 255, 255)

            child_idx = hierarchy[0][child_idx][0]

    threshImg = threshold_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    detectedContours = threshImg.copy()
    detectText = threshImg.copy()
    
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contours_info = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        inTheRect = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                inTheRect = True
                corner_size = 10
                
                is_in_corner = False
                if x <= rx1 + corner_size and y <= ry1 + corner_size:
                    is_in_corner = True
                elif x + w >= rx2 - corner_size and y <= ry1 + corner_size:
                    is_in_corner = True
                elif x <= rx1 + corner_size and y + h >= ry2 - corner_size:
                    is_in_corner = True
                elif x + w >= rx2 - corner_size and y + h >= ry2 - corner_size:
                    is_in_corner = True
                
                if is_in_corner:
                    cv2.drawContours(detectText, [cnt], -1, bg_color, -1)
                break
        
        if inTheRect: continue

        if w > 25 or h > 25:
            parent = hierarchy[0][i][3]
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)
            
            if parent != -1:
                continue
            
            classification, start_point, end_point, approx = classify_contour(
                contours, hierarchy, i
            )

            contours_info.append({
                "type": classification,
                "start": start_point, 
                "end": end_point,
                "approx": approx,
            })
         
    return contours_info, hierarchy, detectedContours, detectText

def merge_nearby_texts(texts, x_margin=10, y_margin=10, dash_width_threshold=5, align_threshold=5, min_vertical_group=5):
    if not texts:
        return []

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

    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))
    lines = []
    current_line = [texts_sorted[0]]
    
    for i in range(1, len(texts_sorted)):
        current_text = texts_sorted[i]
        last_in_line = current_line[-1]
        
        current_top, current_bottom = current_text[1], current_text[1] + current_text[3]
        last_top, last_bottom = last_in_line[1], last_in_line[1] + last_in_line[3]
        
        vertical_overlap = min(current_bottom, last_bottom) - max(current_top, last_top)
        same_line = vertical_overlap > 0 or abs(current_top - last_top) <= y_margin
        
        if same_line:
            current_line.append(current_text)
        else:
            lines.append(current_line)
            current_line = [current_text]
    
    if current_line:
        lines.append(current_line)

    merged_horizontal = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[0])
        current_group = [line_sorted[0]]
        
        for text in line_sorted[1:]:
            last = current_group[-1]
            last_right = last[0] + last[2]
            current_left = text[0]
            
            horizontal_gap = current_left - last_right
            
            if horizontal_gap <= x_margin:
                current_group.append(text)
            else:
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

def processText(texts, y_threshold=5):
    if not texts:
        return ""
    
    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))
    lines = []
    current_line = [texts_sorted[0]]
    
    for i in range(1, len(texts_sorted)):
        current_text = texts_sorted[i]
        prev_text = texts_sorted[i-1]
        
        current_center_y = current_text[1] + current_text[3] / 2
        prev_center_y = prev_text[1] + prev_text[3] / 2
        
        if abs(current_center_y - prev_center_y) <= y_threshold:
            current_line.append(current_text)
        else:
            current_line.sort(key=lambda t: t[0])
            lines.append(current_line)
            current_line = [current_text]
    
    if current_line:
        current_line.sort(key=lambda t: t[0])
        lines.append(current_line)
    
    words = ""
    for line in lines:
        for text in line:
            _, _, _, _, txt = text
            words = words + " " + txt
    
    return words.strip()

def mergeTransition(transitions, texts, margin=5):
    def standardize_point(point):
        if point is None:
            return None
        if isinstance(point, np.ndarray):
            point = point.astype(np.int32)
            if point.ndim > 1:
                point = point.flatten()
            return point
        elif isinstance(point, tuple):
            if all(isinstance(x, (np.integer, np.floating)) for x in point):
                return np.array([float(x) for x in point], dtype=np.float64)
            else:
                return np.array(point, dtype=np.float64)
        elif hasattr(point, '__iter__'):
            return np.array(point, dtype=np.float64)
        else:
            return np.array([point], dtype=np.float64)

    text_to_remove = set()

    for transition in transitions:
        tx1, ty1, tw, th = cv2.boundingRect(transition["approx"])
        tx2, ty2 = tx1 + tw, ty1 + th
        
        trans_bbox_expanded = (tx1 - margin, ty1 - margin, tx2 + margin, ty2 + margin)
        ex1, ey1, ex2, ey2 = trans_bbox_expanded
        
        label = ""
        for text in texts:
            bx, by, bw, bh, text_content = text
            text_bbox = (bx, by, bx + bw, by + bh)
            bx1, by1, bx2, by2 = text_bbox
            
            if (bx1 < ex2 and bx2 > ex1 and by1 < ey2 and by2 > ey1):
                label = text_content
                transition["label"] = label
                text_to_remove.add(text)
                break
    
    texts = [txt for txt in texts if txt not in text_to_remove]
    
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
                if transitions[i].get("start") is None:
                    iPoint1, iPoint2, _ = find_farthest_points(transitions[i].get("approx"))
                else:
                    iPoint1, iPoint2 = transitions[i].get("start"), transitions[i].get("end")
                    arrowhead = iPoint2
                    
                jPoint1, jPoint2 = None, None
                if transitions[j].get("start") is None:
                    jPoint1, jPoint2, _ = find_farthest_points(transitions[j].get("approx"))
                else:
                    jPoint1, jPoint2 = transitions[j].get("start"), transitions[j].get("end")
                    arrowhead = jPoint2
                
                iPoint1 = standardize_point(iPoint1)
                iPoint2 = standardize_point(iPoint2)
                jPoint1 = standardize_point(jPoint1)
                jPoint2 = standardize_point(jPoint2)
                
                if arrowhead is not None:
                    transition_type = "directed arrow"
                    
                    if transitions[i].get("end") is not None:
                        arrowhead = standardize_point(transitions[i]["end"])
                        points = transitions[j].get("approx")
                        max_distance = 0
                        for point in points:
                            point_arr = standardize_point(point)
                            distance = np.linalg.norm(arrowhead - point_arr)
                            if distance > max_distance:
                                max_distance = distance
                                startPoint = point_arr
                    else:
                        arrowhead = standardize_point(transitions[j]["end"])
                        points = transitions[i].get("approx")
                        max_distance = 0
                        for point in points:
                            point_arr = standardize_point(point)
                            distance = np.linalg.norm(arrowhead - point_arr)
                            if distance > max_distance:
                                max_distance = distance
                                startPoint = point_arr
                
                else:
                    transition_type = "line"
                    combined_approx = np.vstack([transitions[i].get("approx"), transitions[j].get("approx")])
                    start, end, _ = find_farthest_points(combined_approx)
                    startPoint = standardize_point(start)
                    arrowhead = standardize_point(end)
                
                merged_approx = np.vstack([transitions[i].get("approx"), transitions[j].get("approx")])
                
                removeTransitions.append(transitions[i])
                removeTransitions.append(transitions[j])
                
                startPoint = standardize_point(startPoint)
                arrowhead = standardize_point(arrowhead)
                
                mergeTransitions.append({
                    "type": transition_type,
                    "start": startPoint,
                    "end": arrowhead,
                    "approx": merged_approx,
                    "label": transitions[i]["label"]
                })
                
                break
    
    for transition in removeTransitions:
        if transition in transitions:
            transitions.remove(transition)
    
    for transition in mergeTransitions:
        transitions.append(transition)
    
    return transitions, texts

def getNodeID(props, point, pos, transition, margin=8):
    allNodes = [props[0], props[1]]
    
    for nodes in allNodes:
        for node in nodes:
            x, y, w, h = node["bbox"]
            px, py = point[0], point[1]
            
            if ((x-margin <= px) and (x+w+margin >= px) and (y-margin <= py) and (y+h+margin > py)):
                node["hasConnection"] = True
                
                if pos not in node:
                    node[pos] = []
                
                node[pos].append({
                    "transitionID": transition.get("id")
                })
                return node["id"]
            
            if node["type"] == "diamond":
                allTextLabel = props[3]
                for textLabel in allTextLabel:
                    tx, ty, tw, th, text = textLabel

                    node_left, node_top, node_right, node_bottom = x, y, x + w, y + h
                    text_left, text_top, text_right, text_bottom = tx, ty, tx + tw, ty + th

                    margin = 30
                    horizontal_dist = max(0, node_left - text_right, text_left - node_right)
                    vertical_dist = max(0, node_top - text_bottom, text_top - node_bottom)
                    distance = (horizontal_dist**2 + vertical_dist**2)**0.5
                    
                    if distance <= margin and props[2] == text:
                        node["hasConnection"] = True
                        
                        if pos not in node:
                            node[pos] = []
                    
                        node[pos].append({
                            "transitionID": transition.get("id")
                        })
                        
                        return node["id"]

    return None

def restructureData(rectangles, contours, texts):
    actionNodes = []
    transitions = []
    otherNotations = []
    
    for rect in rectangles:
        rx, ry, rw, rh = rect
        actionTexts = []
        
        for text in texts:
            tx, ty, tw, th, word = text         
            if ((rx < tx) and (tx + tw < rw) and (ry < ty) and (rh > (ty + th))):
                actionTexts.append(text)
                
        txt = processText(actionTexts)
        actionNodes.append({
            "id": "act" + str(len(actionNodes)+1),
            "type": "ActionNode",
            "bbox": (rx, ry, rw - rx, rh - ry),
            "label": txt
        })
        
        texts = [text for text in texts if text not in actionTexts]
    
    text_merged = merge_nearby_texts(texts)
    
    for contour in contours:          
        if contour["type"] in ["end node", "destruction node", "start node", "diamond node"]:
            classification = contour["type"].split()[0]
            bbox = cv2.boundingRect(contour["approx"])                 
            otherNotations.append({
                "id": "notation" + str(len(otherNotations)+1),
                "type": classification,
                "bbox": bbox,
            })

        elif contour["type"] in ["line", "directed arrow"]:
            transitions.append(contour)
        
    transitions, text_merged = mergeTransition(transitions, text_merged)
    
    for i, transition in enumerate(transitions): 
        transition["id"] = "line" + str(i + 1)
         
    for transition in transitions:
        if transition.get("type") == "directed arrow":
            start, end = transition.get("start"), transition.get("end")
            
            props = (actionNodes, otherNotations, transition.get("label"), text_merged)
            
            startingNodeID = getNodeID(props, start, "from", transition)
            destinationNodeID = getNodeID(props, end, "to", transition)

    structuredData = {
        "nodes": [],
        "transitions": []
    }
    
    global actors, removeVerticalLines, removeHorizontalLines
      
    if len(removeVerticalLines) > 0:
        removeVerticalLines.sort(key=lambda line: line[0])
        removeHorizontalLines.sort(key=lambda line: line[1]) 
        
        for i in range(len(removeVerticalLines) - 1):
            line1 = removeVerticalLines[i]
            line2 = removeVerticalLines[i + 1]
            
            actorName = ""
            
            vx1, vy1, vw1, vh1 = line1
            vx2, vy2, vw2, vh2 = line2
            
            left_boundary = vx1 + vw1
            right_boundary = vx2
            
            if len(removeHorizontalLines) >= 2:
                hline1 = removeHorizontalLines[0]
                hline2 = removeHorizontalLines[1]
                
                hx1, hy1, hw1, hh1 = hline1
                hx2, hy2, hw2, hh2 = hline2
                
                top_boundary = hy1 + hh1
                bottom_boundary = hy2
                
                found_texts = []
                for text in text_merged:
                    tx, ty, tw, th, text_content = text
                    
                    text_right = tx + tw
                    text_bottom = ty + th
                    
                    vertical_inside = (tx >= left_boundary and text_right <= right_boundary)
                    horizontal_inside = (ty >= top_boundary and text_bottom <= bottom_boundary)
                    
                    if vertical_inside and horizontal_inside:
                        found_texts.append(text_content)
                
                if found_texts:
                    found_texts_sorted = sorted(found_texts, key=lambda t: (ty, tx))
                    actorName = " ".join(found_texts_sorted)
                    text_merged[:] = [t for t in text_merged if t[4] not in found_texts]
            
            actors.append({
                "label": actorName,
                "x_boundaries": [left_boundary, right_boundary]
            })

    def getActor(node):
        bbox = [int(x) for x in node["bbox"]]
        
        for actor in actors:
            actorBoundaries = actor["x_boundaries"]
            if actorBoundaries[0] < bbox[0] and actorBoundaries[1] > bbox[0] + bbox[3]:
                return actor["label"]
        return None
    
    for node in actionNodes:
        structuredData["nodes"].append({
            "id": node.get("id"),
            "type": node.get("type"),
            "bbox": [int(x) for x in node["bbox"]],
            "label": node.get("label"),
            "from": node.get("from", None),
            "to": node.get("to", None),
            "actor": getActor(node) if len(actors) > 1 else None
        })
    
    for notation in otherNotations:
        structuredData["nodes"].append({
            "id": notation["id"],
            "type": notation["type"],
            "bbox": [int(x) for x in notation["bbox"]],
            "from": notation.get("from", None),
            "to": notation.get("to", None),
            "actor": getActor(notation) if len(actors) > 1 else None
        })
    
    for transition in transitions:
        transition_data = {
            "id": transition["id"],
            "type": transition["type"],
            "start": [int(transition["start"][0]), int(transition["start"][1])] if transition["start"] is not None else None,
            "end": [int(transition["end"][0]), int(transition["end"][1])] if transition["end"] is not None else None,
        }
        if "label" in transition:
            transition_data["label"] = transition["label"]
            
        structuredData["transitions"].append(transition_data)
    
    return structuredData 

def process_activity_diagram(file):
    
    start_time = time.time()
    
    """Main function to process activity diagram - called from Flask"""
    image, gray = load_image(file)
    thresh = threshold_image(gray)
    vertical_lines, horizontal_lines = extract_lines(thresh)
    
    vertical_segments = get_segments(vertical_lines)
    horizontal_segments = get_segments(horizontal_lines)
    
    vertical_pairs, image = find_vertical_pairs(vertical_segments, image)
    horizontal_pairs, image = find_horizontal_pairs(horizontal_segments, image)
    
    rectangles, lines_to_remove = detect_rectangles(vertical_pairs, horizontal_pairs, vertical_segments, horizontal_segments)
    image_rect = remove_rectangles(image, lines_to_remove, vertical_segments, horizontal_segments)
    
    noRectImg = image_rect.copy()
    
    contours, hierarchy, detectedContours, textImg = detect_contours(noRectImg, rectangles)
    
    textData = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    
    texts = []
    for i in range(len(textData['text'])):
        text = textData['text'][i].strip()
        if text != "":
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