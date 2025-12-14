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
        # print("Both wala")
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
        
        # print("Distances: ", distances, " Max: ", max_dist, " Min: ", min_dist, " Avg: ", avg_distance, " Scaled Min: ", scaled_min, " Dynamic Tol: ", dynamic_tolerance)
        
        if max(distances) - min(distances) <= dynamic_tolerance+5:
            seq_np = np.array(seq_points, dtype=np.int32)
            hull = cv2.convexHull(seq_np, returnPoints=True)
            hull_count = len(hull)
            # print("NSEq ",len(seq_points))
            # print("Hull ", hull_count)
            # print("Found! Convex Hull Points: ", hull_count)
            # print("Points: ", seq_points)
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
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Check all child contours for circle shape
    child_idx = hierarchy[0][idx][2]
    while child_idx != -1:
        child_cnt = contours[child_idx]
        
        # Calculate circularity of child contour
        child_area = cv2.contourArea(child_cnt)
        child_perimeter = cv2.arcLength(child_cnt, True)
        
        if child_perimeter > 0:
            circularity = 4 * np.pi * child_area / (child_perimeter * child_perimeter)
        else:
            circularity = 0
        
        # Check if child is circular (typical circle has circularity close to 1)
        if circularity > 0.7:  # High circularity indicates circle
            # print(f"Found circular child - circularity: {circularity:.2f}")
            return "actor", None, None, cnt
        
        # Move to next sibling child
        child_idx = hierarchy[0][child_idx][0]
    
    # If no circular child found, check for other child-based classifications
    return "unknown", None, None, cnt
    
   
        
def classify_contour(contours, hierarchy, idx):
    
    global epsilon_ratio
    
    epsilon_ratio = 0.04
    cnt = contours[idx]
    has_five_seq, convexHull = None, None
    approx = None
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    bbox_area = w * h
    
    child_idx = hierarchy[0][idx][2]
    
    if child_idx != -1  and h > 100:
        return classify_based_on_child(contours, hierarchy, idx)
    
    
    # Calculate circularity (for start/end nodes)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    else:
        circularity = 0
    
    # Check for circular shape (start node)
    if bbox_area > 0 and (area / bbox_area) >= 0.75 and circularity >= 0.7 and h > 50:
      
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)   
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        return "start node", None, None, approx

    bottom_points = []
    
    # likey arrow
    while True:
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)   
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        bottom_points = []
        
        # delete message, X notation   
        if len(approx) == 8 and h/w > 0.8:
            return "delete", None, None, approx
        elif h > 50 and len(approx) > 5:
            # check for 5 point sequence in the left most bottom corner
            points = np.squeeze(approx)    
            
            half_height_y = y + h//2
            
            # filter points
            bottom_points = []
            for point in points: 
                px, py = point
                if py > half_height_y:
                    bottom_points.append(point)           
        
        farthest_points = None
        
        if(len(bottom_points) >= 5):
            bottom_approx = np.array(bottom_points).reshape(-1, 1, 2)
            farthest_points = find_farthest_points(bottom_approx)
        else :
             # find two farthest points
            farthest_points = find_farthest_points(approx)   
        
        if farthest_points is None:
            return "Unknown", None, None, approx
        
        point1, point2, max_distance = farthest_points 
        # check if the approx has 5points sequence 
        # if(epsilon_ratio > 0.01): 
        #     epsilon_ratio = epsilon_ratio / 2 
        #     continue
        
        has_five_seq, convexHull = find_five_point_sequence(approx, point1, point2)
        if has_five_seq and h < 50: #break if have 5-point sequence,
         
            break
            # 0.02 to get the self message
        elif has_five_seq and len(bottom_points) > 5:
            break
        
        elif epsilon_ratio < 0.0025:
            break

        epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx

    if has_five_seq:
        if bottom_points and convexHull == 3:
            classification = "self"
            start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
            return classification, start_point, end_point, approx
        
        elif convexHull and convexHull == 3:
        
            classification = "asynchronous"
            start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
            
            return classification, start_point, end_point, approx
        elif convexHull and convexHull <= 5:
           
            classification = "synchronous"
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
    
    dashLineArr = []
    
    # find all the contours that are inside the rectangles and remove/skip them
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # check if the contour is inside the rect
        r = None
        loc = None
        inTheRect = False
        is_in_corner = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            r = rect    
            # Check if contour bbox is within rectangle bbox
            if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                # Fill rectangle area with background color to remove it
                # inTheRect = True
                # check if the contour is in the corner of the bounding box
                # Define corner size (adjust based on your needs)
                corner_size = 5  # pixels
                
                # Check if contour is in any of the four corners
                is_in_corner = False
                
                # Top-left corner check
                if x <= rx1 + corner_size and y <= ry1 + corner_size:
                    is_in_corner = True
                    loc = "top-left"
                # Top-right corner check  
                elif x + w >= rx2 - corner_size and y <= ry1 + corner_size:
                    is_in_corner = True
                    loc = "top-right"
                # Bottom-left corner check
                elif x <= rx1 + corner_size and y + h >= ry2 - corner_size:
                    is_in_corner = True
                    loc = "bottom left"
                # Bottom-right corner check
                elif x + w >= rx2 - corner_size and y + h >= ry2 - corner_size:
                    is_in_corner = True
                    loc = "bottom-right"
                
                if is_in_corner:
                    cv2.rectangle(detectText, (x, y), (x + w, y + h), bg_color, -1)  # Fill the whole bounding box
                    # cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
            
                break  # No need to check other rectangles
            
        if inTheRect and (w < 20 or h < 20): continue  # Skip this contour entirely
        
        if(is_in_corner): continue
        
        if (w > 15 or h > 15) and w+2 >= h or h > 50:  # filter small noise
            
            parent = hierarchy[0][i][3]
            
            if parent != -1:
                continue  # skip child contours
            
            count += 1
            
            classification, start_point, end_point, approx = classify_contour(
                contours, hierarchy, i
            )
            
            x, y, w, h = cv2.boundingRect(approx)

            # newMessage = {
            #     "type": classification,
            #     "start" : start_point, 
            #     "end": end_point,
            #     "approx": approx,
            # }
            # print("Hello:", classification)
            # if(classification == "self"):
            #     print("Last startPoints", start_point)
            #     print("new startPoints", end_point[0], np.int32(y))
            #     newMessage["start"] = [end_point[0], np.int32(y)]
            #     contours_info.append(newMessage)
            #     continue
            
            # get dashline
            print("classification:", classification, " w:", w, " h:", h)
            if (w < 40 or ((classification == "asynchronous" or classification == "synchronous") and w < 100)) and classification != "self":
                added = False
                for group in dashLineArr:
                    gx, gy, gw, gh = cv2.boundingRect(group[0])
                    
                    if abs(y - gy) < 20 : # same horizontal level
                        group.append(approx)
                        added = True
                    
                        break
                    
                  # If not added to any group, create a new one
                if not added:
               
                    dashLineArr.append([cnt])

                continue  # Skip further processing for dash contours
            
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
                
            # Draw bounding box
            cv2.rectangle(contour_img, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(contour_img, str(count) + " " + str(classification), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # # Draw approx points
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            newMessage = {
                "type": classification,
                "start" : start_point, 
                "end": end_point,
                "approx": approx,
            }
            # print("Hello:", classification)
            # if(classification == "self"):
            #     print("Last startPoints", start_point)
            #     print("new startPoints", end_point[0], np.int32(y))
            #     newMessage["start"] = [end_point[0], np.int32(y)]
            contours_info.append(newMessage)
            
        
            for pt in approx:
                cv2.circle(contour_img, tuple(pt[0]), 1, (255, 0, 0), -1)


    # print("58: ", contours_info)
    # print("Dash line groups details:", len(dashLineArr))
    c = 0
    loopContour = set()
    altContour = set()
    
    # NEW: Process dash line groups to determine start/end for reply messages
    for group in dashLineArr:
        # Flatten all points from contours in the group
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in group])  # shape: (N, 2)
        
        if(len(group) < 5): 
            continue
        
        cv2.drawContours(detectText, [all_points], -1, bg_color, -1)  # Fill contour with background color
        
        # Calculate overall bounding box from all points
        x, y, w, h = cv2.boundingRect(all_points)
        group_min_x, group_min_y = x, y
        group_max_x, group_max_y = x + w, y + h
        
        isDashLineSeparator = False
        
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            isDashLineSeparator = False
            
            if(abs(x2 - x1) < 50) and (y2-y1) < 200: continue
            if(group_min_x <= x1 + 20 and group_max_x >= x2-20):                
                isDashLineSeparator = True
                altContour.add(rect)
                break
            
        if isDashLineSeparator:
            continue
        
        # Sort group by X coordinate to ensure left-to-right order
        group_sorted = sorted(group, key=lambda cnt: cv2.boundingRect(cnt)[0])

        # Check first and last contours for arrowheads (>4 points)
        first_contour = group_sorted[0]
        last_contour = group_sorted[-1]

        first_approx_points = len(first_contour)
        last_approx_points = len(last_contour)

        # print(f"Dash Group {c+1}: First contour points: {first_approx_points}, Last contour points: {last_approx_points}")

        # Determine start and end based on arrowhead position
        if last_approx_points > 4:
            start_contour = first_contour
            end_contour = last_contour
            direction = "left_to_right"
        elif first_approx_points > 4:
            start_contour = last_contour
            end_contour = first_contour
            direction = "right_to_left"
        else:
            start_contour = first_contour
            end_contour = last_contour
            direction = "unknown"

        # Get start and end points from the determined contours
        start_x, start_y, start_w, start_h = cv2.boundingRect(start_contour)
        end_x, end_y, end_w, end_h = cv2.boundingRect(end_contour)

        # Calculate actual start and end coordinates
        if direction == "left_to_right":
            start_point = (start_x, start_y + start_h//2)
            end_point = (end_x + end_w, end_y + end_h//2)
        else:  # right_to_left or unknown
            start_point = (start_x + start_w, start_y + start_h//2)
            end_point = (end_x, end_y + end_h//2)

        c += 1

        # Add as reply message with all points stored in "approx"
        contours_info.append({
            "type": "reply",
            "start": start_point,
            "end": end_point,
            "direction": direction,
            "segment_count": len(group),
            "approx": all_points  # all dash line points
        })

        # Draw bounding box for the group
        cv2.rectangle(contour_img, (group_min_x, group_min_y), (group_max_x, group_max_y), (0, 255, 255), 2)

        # Draw direction arrow
        cv2.arrowedLine(contour_img, start_point, end_point, (255, 0, 255), 2, tipLength=0.05)

        cv2.putText(contour_img, f"Reply {c} ({direction})", 
                    (group_min_x, group_min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)

    for rect in rectangles:
        x1, y1, x2, y2 = rect

        if (x2-x1 < 100): continue

        if rect not in altContour:
            print(x1, " ", y1, " " , x2, " ", y2)
            loopContour.add(rect)
            
    for rect in altContour:
        x1, y1, x2, y2 = rect
    
        # Create proper contour array from rectangle coordinates
        contour_points = np.array([
            [[x1, y1]],     # Top-left
            [[x2, y1]],     # Top-right  
            [[x2, y2]],     # Bottom-right
            [[x1, y2]]      # Bottom-left
        ], dtype=np.int32)
        
        contours_info.append({
            "type": "alt",
            "start": None, 
            "end": None,
            "approx": contour_points,  # Proper contour array
        })
        cv2.rectangle(contour_img, (x1, y1), (x2, y2), (0, 0, 255), 1)        
            
    for rect in loopContour:
        
        x1, y1, x2, y2 = rect
        # Create proper contour array from rectangle coordinates
        contour_points = np.array([
            [[x1, y1]],     # Top-left
            [[x2, y1]],     # Top-right  
            [[x2, y2]],     # Bottom-right
            [[x1, y2]]      # Bottom-left
        ], dtype=np.int32)
        
        contours_info.append({
            "type": "loop",
            "start": None, 
            "end": None,
            "approx": contour_points,  # Proper contour array
        })
        
        cv2.rectangle(contour_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    
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


def is_text_near_rect(tx, ty, tw, th, rect_x, rect_y, rect_w, rect_h, max_dist=10):
    # center of text
    cx = tx + tw / 2
    cy = ty + th / 2

    # center of rect
    rx = rect_x + rect_w / 2
    ry = rect_y + rect_h / 2

    # distance between centers
    dist_x = max(rect_x - (tx + tw), tx - (rect_x + rect_w), 0)
    dist_y = max(rect_y - (ty + th), ty - (rect_y + rect_h), 0)

    # if distance is smaller than max_dist in either direction, consider near
    return dist_x <= max_dist and dist_y <= max_dist


def extract_text_in_region(rect, texts, contourType):
    matched_text = None
    matched_entry = None
    
    if contourType == "obj":
        x, y, w, h = rect
        for entry in texts:
            tx, ty, tw, th, content = entry
            tx, ty, tw, th = int(tx), int(ty), int(tw), int(th)
            if tx >= x and tx + tw <= x + w and ty >= y and ty + th <= y + h:
                matched_text = content
                matched_entry = entry
                break  # stop at first found

        # Remove matched entry
        if matched_entry and matched_entry in texts:
            texts.remove(matched_entry)

        return matched_text

    elif contourType == "message" or contourType == "notation":
        # rect["approx"] contains all points of the line or loop
        all_points = rect["approx"]  # np.array of shape (N, 2)

        # Compute bounding box of all points
        x, y, w, h = cv2.boundingRect(all_points)

        matched_texts = []
    
        for entry in texts[:]:
            tx, ty, tw, th, content = entry
            tx, ty, tw, th = int(tx), int(ty), int(tw), int(th)

            inside_box = (tx >= x and tx + tw <= x + w and ty >= y and ty + th <= y + h)
            near_text = is_text_near_rect(tx, ty, tw, th, x, y, w, h, max_dist=15)

            if inside_box or near_text:
                matched_texts.append(content)
                texts.remove(entry)

        # Sort left-to-right by x coordinate
        matched_texts.sort(key=lambda t: t[0] if isinstance(t, tuple) else 0)

        # Join into a single string
        return " ".join(matched_texts) if matched_texts else None

    elif contourType == "alt":
        # Get alt contour points and bounding box
        all_points = rect["approx"]
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Find all texts inside or near the alt contour
        all_alt_texts = []
        
        for entry in texts[:]:
            tx, ty, tw, th, content = entry
            tx, ty, tw, th = int(tx), int(ty), int(tw), int(th)

            inside_box = (tx >= x and tx + tw <= x + w and ty >= y and ty + th <= y + h)
            near_text = is_text_near_rect(tx, ty, tw, th, x, y, w, h, max_dist=15)

            if inside_box or near_text:
                all_alt_texts.append((tx, ty, tw, th, content))
        
        if not all_alt_texts:
            return {"if_condition": "", "else_condition": ""}
        
        # Sort all texts by y-coordinate (top to bottom)
        all_alt_texts.sort(key=lambda t: (t[1], t[0]))
        
        # Group texts that are close together (≤5px gap horizontally AND vertically)
        text_groups = []
        current_group = [all_alt_texts[0]]
        
        for i in range(1, len(all_alt_texts)):
            current_text = all_alt_texts[i]
            last_text = current_group[-1]
            
            # Calculate horizontal and vertical gaps
            horizontal_gap = current_text[0] - (last_text[0] + last_text[2])
            vertical_gap = abs(current_text[1] - last_text[1])
            
            # Texts are grouped if they're on roughly the same line and close horizontally
            if horizontal_gap <= 5 and vertical_gap <= 10:
                current_group.append(current_text)
            else:
                text_groups.append(current_group)
                current_group = [current_text]
        
        if current_group:
            text_groups.append(current_group)
        
        # Find groups in different sections of the alt contour
        top_left_group = None
        middle_left_group = None
        
        # Calculate section boundaries for alt contour
        top_section_end = y + h * 0.3    # Top 30% - for IF condition
        middle_section_start = y + h * 0.4  # Middle 40-70% - for ELSE condition
        middle_section_end = y + h * 0.7
        left_boundary = x + w * 0.3      # Left 30% - for both conditions
        
        for group in text_groups:
            # Calculate group's position
            group_x_min = min(text[0] for text in group)
            group_y_min = min(text[1] for text in group)
            group_y_max = max(text[1] + text[3] for text in group)
            group_y_center = (group_y_min + group_y_max) / 2
            
            # Check if group is in left side
            is_in_left_side = group_x_min <= left_boundary
            
            if not is_in_left_side:
                continue  # Skip groups not in left side
            
            # Check if group is in top-left section (IF condition)
            if group_y_center <= top_section_end and top_left_group is None:
                top_left_group = group
            
            # Check if group is in middle-left section (ELSE condition)  
            elif (middle_section_start <= group_y_center <= middle_section_end 
                  and middle_left_group is None):
                middle_left_group = group
        
        # Remove detected groups from global texts and prepare return value
        result = {"if_condition": "", "else_condition": ""}
        
        # Process IF condition (top-left group)
        if top_left_group:
            # Remove from global texts
            for text_entry in top_left_group:
                tx, ty, tw, th, content = text_entry
                for entry in texts[:]:
                    if (entry[0] == tx and entry[1] == ty and 
                        entry[2] == tw and entry[3] == th and 
                        entry[4] == content):
                        texts.remove(entry)
                        break
            
            text_contents = [text[4] for text in top_left_group]
            result["if_condition"] = " ".join(text_contents)
        
        # Process ELSE condition (middle-left group)
        if middle_left_group:
            # Remove from global texts
            for text_entry in middle_left_group:
                tx, ty, tw, th, content = text_entry
                for entry in texts[:]:
                    if (entry[0] == tx and entry[1] == ty and 
                        entry[2] == tw and entry[3] == th and 
                        entry[4] == content):
                        texts.remove(entry)
                        break
            
            text_contents = [text[4] for text in middle_left_group]
            result["else_condition"] = " ".join(text_contents)
        
        return result
         
    elif contourType == "loop":
        # Get loop contour points and bounding box
        all_points = rect["approx"]
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Find all texts inside or near the loop contour
        all_loop_texts = []
        
        for entry in texts[:]:
            tx, ty, tw, th, content = entry
            tx, ty, tw, th = int(tx), int(ty), int(tw), int(th)

            inside_box = (tx >= x and tx + tw <= x + w and ty >= y and ty + th <= y + h)
            near_text = is_text_near_rect(tx, ty, tw, th, x, y, w, h, max_dist=15)

            if inside_box or near_text:
                all_loop_texts.append((tx, ty, tw, th, content))
        
        if not all_loop_texts:
            return None
        
        # Sort all texts by y-coordinate first (top to bottom), then by x (left to right)
        all_loop_texts.sort(key=lambda t: (t[1], t[0]))
        
        # Group texts that are close together (≤5px gap horizontally AND vertically)
        text_groups = []
        current_group = [all_loop_texts[0]]
        
        for i in range(1, len(all_loop_texts)):
            current_text = all_loop_texts[i]
            last_text = current_group[-1]
            
            # Calculate horizontal and vertical gaps
            horizontal_gap = current_text[0] - (last_text[0] + last_text[2])
            vertical_gap = abs(current_text[1] - last_text[1])
            
            # Texts are grouped if they're on roughly the same line and close horizontally
            if horizontal_gap <= 5 and vertical_gap <= 10:
                current_group.append(current_text)
            else:
                # Start a new group
                text_groups.append(current_group)
                current_group = [current_text]
        
        # Add the last group
        if current_group:
            text_groups.append(current_group)
        
        # Find the group that's near the left boundary
        left_group = None
        left_boundary = x + w * 0.2  # Left 20% of the loop contour
        
        for group in text_groups:
            # Get the leftmost text in the group
            leftmost_text = min(group, key=lambda t: t[0])
            leftmost_x = leftmost_text[0]
            
            # Check if this group is near left boundary (within 20% of width from left)
            if (leftmost_x - x) <= left_boundary:
                left_group = group
                break
        
        # If no group found near left boundary, use the first group as fallback
        if not left_group and text_groups:
            left_group = text_groups[0]
        
        # Return and remove the left group
        if left_group:
            # Remove only the left group texts from global texts list
            for text_entry in left_group:
                tx, ty, tw, th, content = text_entry
                # Find and remove the exact entry from texts
                for entry in texts[:]:
                    if (entry[0] == tx and entry[1] == ty and 
                        entry[2] == tw and entry[3] == th and 
                        entry[4] == content):
                        texts.remove(entry)
                        break
            
            # Return the joined text from left group
            text_contents = [text[4] for text in left_group]
            return " ".join(text_contents)
        else:
            # No group found
            return None
        
    return matched_text


def restructureData(rectangles, contours, texts):
    objectNodes = []
    messages = []
    notations = []
    # ids
    objectID = 0 
    messageID = 0
    notationID = 0
    text = ""
    hasDeleteMessage = False
    
    print("Text: ", len(texts))
    for cont in contours:   
        classification = cont["type"]
        startPoint = cont["start"]
        endPoint = cont["end"]
        text = ""
        
        if (classification == "asynchronous" or classification == "synchronous" or classification == "self" or classification == "reply"):
        
            text = extract_text_in_region(cont, texts, "message")
            messages.append({
                "messageID": "msg" + str(messageID),
                "type": classification,
                "start": startPoint,
                "end": endPoint,
                "label": text,
            })
        
            messageID += 1
            
            print(classification, " : ", text)
        
        else :
            # if classification == "alt" or classification == "loop": continue
            x, y, w, h = cv2.boundingRect(cont["approx"])
            text = None
            
            if(classification == "alt" or classification == "loop"):
               
                if classification == "alt":    
                    conditions = extract_text_in_region(cont, texts, classification)
                    ifCondition = None
                    elseCondition = None
                    if(conditions):
                        ifCondition = conditions.get("if_condition")
                        elseCondition = conditions.get("else_condition")
                    else:
                        ifCondition = ""
                        elseCondition = ""
                    print("If ", ifCondition)
                    print("Else ", elseCondition)
                    notations.append({
                        "notationID": "notation" + str(notationID),
                        "type": classification,
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "ifCondition": ifCondition,
                        "elseCondition": elseCondition,
                    })
                    print()
                    
                else:
                    # continue
                    # check if the contour has message
                    isALoop = False
                    
                    for msgContour in contours:
                        if(msgContour is cont): continue
                        if msgContour["type"] in {"loop", "obj", "alt", "delete", "start node", "actor"} :
                            continue
                        print("===================")
                        mx, my, mw, mh = cv2.boundingRect(msgContour["approx"])
                        print(mx, " ", my, " ", mw, " ", mh)
                        print(x, " ", y, " ", w, " ", h)
                        
                        print("===================")
                        # print("type ", msgContour["type"])
                        if(mx > x) and (mw+mx < x+w):
                            isALoop = True                     
                            break
                
                    notationType = ""
                    label = ""
                    if isALoop: 
                        notationType = "loop"
                        label = extract_text_in_region(cont, texts, notationType)  # ✅ Reusable call
                    else: 
                        notationType = "obj"
                        label = extract_text_in_region((x,y,w,h), texts, notationType)  # ✅ Reusable call
                    print()
                    print("ID: ", notationID)
                    print("Classification: 696", notationType)
                    print("Label: 670", label)
                  
                    notations.append({
                        "notationID": "notation" + str(notationID),
                        "type": notationType,
                        "bbox": {"x": x, "y": y, "w": w, "h": h},
                        "label": label  # None if no label found
                    })         

            else:
                text = extract_text_in_region(cont, texts, "notation")             
                notations.append({
                    "notationID": "notation" + str(notationID),
                    "type": classification,
                    "bbox": {"x": x, "y": y, "w": w, "h": h},
                    "label": text,
                })
            
            notationID += 1
            print(classification, " : ", text)

    print(len(texts))
    
    print("Msg: ", len(messages))
    print("Notation: ", len(notations))
    # return
    for msg in messages:
        start_point = msg["start"]
        end_point = msg["end"]

        sender = None
        receiver = None

        if start_point is not None:
            sender = find_nearby_notation(start_point, notations)  # Check if pointing to X/delete node
        if end_point is not None:
            receiver = find_nearby_notation(end_point, notations)  # Check if pointing to X/delete node

        # If not found, check objects - only if points are not None
        if sender is None and start_point is not None:
            sender = find_nearby_object(start_point, objectNodes)  # Fall back to regular objects
            
        if receiver is None and end_point is not None:
            receiver = find_nearby_object(end_point, objectNodes)  # Fall back to regular objects

        if sender is None and msg["type"] == "self":
            sender = receiver
        
        print(msg["type"])
        print("Start point:", start_point, "-> sender:", sender , )
        print("End point:", end_point, "-> receiver:", receiver)
        print(" message ",msg["label"])
        print()
        
        msg["from"] = sender
        msg["to"] = receiver
    
    # find parent of delete node
    for notation in notations:
        if notation["type"] == "delete":
            # Get the delete notation's bounding box
            nx, ny, nw, nh = notation["bbox"]["x"], notation["bbox"]["y"], notation["bbox"]["w"], notation["bbox"]["h"]
            
            # Look for parent object in the same notations array
            parent_object = None
            for obj in notations:
                # Skip if it's not an object or if it's the same delete notation
                print("Type ", obj["type"])
                if obj["type"] == "obj":
                    print("hello2")
                    ox, oy, ow, oh = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["w"], obj["bbox"]["h"]
                    
                    # Check if delete notation is inside this object's bounding box
                    margin = 5
                    print(nx, " ", ny, " ", nw, " ", nh)
                    print(ox, " ", oy, " ", ow, " ", oh)
                    if (ox <= nx and 
                        nx + nw <= ow+ox):
                        parent_object = obj
                        break
                    
            # Add parentID to the delete notation
            if parent_object:
                notation["parentID"] = str(parent_object["notationID"])
            else:
                notation["parentID"] = None
    
    result = {
        "notations": [],
        "messages": []
    }
    
    # Add all notations (objects, loops, alt, delete, etc.)
    for notation in notations:
        notation_data = {
            "id": notation["notationID"],
            "type": notation["type"],
            "bbox": notation["bbox"],
            "x": notation["bbox"]["x"],
            "y": notation["bbox"]["y"],
            "width": notation["bbox"]["w"],
            "height": notation["bbox"]["h"]
        }
        
        # Add type-specific fields
        if notation["type"] == "alt":
            notation_data["ifCondition"] = notation.get("ifCondition", "")
            notation_data["elseCondition"] = notation.get("elseCondition", "")
        elif notation["type"] == "loop":
            notation_data["label"] = notation.get("label", "")
        elif notation["type"] == "obj":
            notation_data["label"] = notation.get("label", "")
        elif notation["type"] == "delete":
            notation_data["parentID"] = notation.get("parentID")
            notation_data["label"] = "X"
        elif notation["type"] in ["start node", "actor"]:
            notation_data["label"] = notation.get("label", "")
        
        result["notations"].append(notation_data)
    
    # Add all messages
    for msg in messages:
        result["messages"].append({
            "id": msg["messageID"],
            "type": msg["type"],
            "from": msg.get("from"),
            "to": msg.get("to"),
            "label": msg.get("label", ""),
            "startPoint": msg.get("start"),
            "endPoint": msg.get("end")
        })
        
         # === PASTE THE BLOCK RIGHT HERE ===
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj

    # Apply conversion to the entire result structure
    result = convert_numpy_types(result)
    # === END OF PASTED BLOCK ===
    
    return result
    

def find_nearby_object(point, objects, margin=5):
    px, py = point
    for obj in objects:
        if(obj["type"] != "obj"): 
            continue
        x, y, w, h = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["w"], obj["bbox"]["h"]
        # check if point is inside box with margin
        if (px >= x - margin) and (px <= x + w + margin) :
            return str(obj["notationID"])
    return None


def find_nearby_notation(point, notations, margin=5):
    px, py = point
    for note in notations:
        if(note["type"] == "alt" or note["type"] == "loop"): continue
        x, y, w, h = note["bbox"]["x"], note["bbox"]["y"], note["bbox"]["w"], note["bbox"]["h"]
        if (px >= x - margin) and (px <= x + w + margin):
            return (str(note["notationID"]))
    return None


def merge_nearby_texts(texts, x_margin=10, y_margin=5, dash_width_threshold=5, align_threshold=5, min_vertical_group=5):
    """
    Merge nearby text boxes into sentences while ignoring vertical dash-like noise.
    
    :param texts: list of tuples (x, y, w, h, text)
    :param x_margin: horizontal gap threshold to merge
    :param y_margin: vertical gap threshold to consider same line
    :param dash_width_threshold: max width to consider as dash noise
    :param align_threshold: max allowed difference in X to consider aligned
    :param min_vertical_group: number of aligned small-width texts to be considered noise
    :return: merged list of tuples (x, y, w, h, sentence)
    """
    if not texts:
        return []

    # --- STEP 1: Detect and remove vertical dash groups ---
    dash_candidates = [t for t in texts if t[2] < dash_width_threshold]  # width < threshold

    remove_set = set()
    # Group by close X alignment
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

    # Filter valid texts
    texts = [t for t in texts if t not in remove_set]

    # --- STEP 2: Normal merging process ---
    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))

    merged_texts = []
    if not texts_sorted:
        return merged_texts

    current_group = [texts_sorted[0]]

    for entry in texts_sorted[1:]:
        x, y, w, h, text = entry
        last_x, last_y, last_w, last_h, last_text = current_group[-1]

        same_line = abs(y - last_y) <= y_margin
        close_horizontally = (x - (last_x + last_w)) <= x_margin

        if same_line and close_horizontally:
            current_group.append(entry)
        else:
            x0 = min(e[0] for e in current_group)
            y0 = min(e[1] for e in current_group)
            x1 = max(e[0] + e[2] for e in current_group)
            y1 = max(e[1] + e[3] for e in current_group)
            sentence = " ".join(e[4] for e in current_group)
            merged_texts.append((x0, y0, x1 - x0, y1 - y0, sentence))
            current_group = [entry]

    if current_group:
        x0 = min(e[0] for e in current_group)
        y0 = min(e[1] for e in current_group)
        x1 = max(e[0] + e[2] for e in current_group)
        y1 = max(e[1] + e[3] for e in current_group)
        sentence = " ".join(e[4] for e in current_group)
        merged_texts.append((x0, y0, x1 - x0, y1 - y0, sentence))

    return merged_texts


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
    
    for cont in contours:
        print("Type:", cont["type"])
    
    # # --- 3. OCR + bounding box info ---
    # textData = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    # d1 = pytesseract.image_to_string(textImg)
    # # --- 4. Convert grayscale to BGR for drawing ---
    # textImg = cv2.cvtColor(textImg, cv2.COLOR_GRAY2BGR)
    # print("Contour ", len(contours))
    # # --- 5. Draw bounding boxes ---
    # n_boxes = len(textData['level'])
    # for i in range(n_boxes):
    #     text = textData['text'][i].strip()
    #     conf = int(textData['conf'][i])

    #     if text != "":  # filter low confidence or empty
    #         (x, y, w, h) = (textData['left'][i], textData['top'][i], textData['width'][i], textData['height'][i])
    #         cv2.rectangle(textImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(textImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # # ✅ Convert pytesseract dict into clean list format
    # texts = []
    # for i in range(len(textData['text'])):
    #     text = textData['text'][i].strip()
    #     if text != "":  # ignore empty OCR outputs
    #         x = int(textData['left'][i])
    #         y = int(textData['top'][i])
    #         w = int(textData['width'][i])
    #         h = int(textData['height'][i])
    #         texts.append((x, y, w, h, text))
    
    # # Merge nearby words into sentences
    # texts_merged = merge_nearby_texts(texts)
    
    # result = restructureData(rectangles, contours, texts_merged)
    
    # # print(result)
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    # # visualize(thresh, drawContour, image2)
    visualize(image, drawContour, textImg)


# ---------------- Run ----------------
main("./images/sqWithLoop.png")
