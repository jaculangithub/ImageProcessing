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
            print("NSEq ",len(seq_points))
            print("Hull ", hull_count)
            print("Found! Convex Hull Points: ", hull_count)
            print("Points: ", seq_points)
            if hull_count >= 4:
                return seq_points, hull_count
            elif hull_count == 3:
                return seq_points, hull_count   

    # print("No 5points found that don't contain both farthest points")
    return None, None


# def classify_based_on_child(contours, hierarchy, idx):
#     """
#     Classify contour based on its child contour for activity diagrams
#     """
    
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
            print(f"Found circular child - circularity: {circularity:.2f}")
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
        print("Probably actor")
        return classify_based_on_child(contours, hierarchy, idx)
    
    
    # Calculate circularity (for start/end nodes)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    else:
        circularity = 0
    
    print(" area: ", area, "B area: ", bbox_area, " Circularity ", circularity)
    # Check for circular shape (start node)
    if bbox_area > 0 and (area / bbox_area) >= 0.75 and circularity >= 0.7 and h > 50:
        print("start node")
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
        if len(approx) == 8 and w/h > 0.9:
            return "delete", None, None, approx
        elif h > 50 and len(approx) > 5:
            # check for 5 point sequence in the left most botto corner, how to do it?
            # ge
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
        # print("has 5", len(has_five_seq))
        print("First 5point ", convexHull)
        if has_five_seq and h < 50: #break if have 5-point sequence,
            print("meron likely an arrow 5 points")
            print("Convex hull ", convexHull)
            break
            # 0.02 to get the self message
        elif has_five_seq and len(bottom_points) > 5:
            break
        
        elif epsilon_ratio < 0.0025:
            print("di arrow")
            break

        epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx

    if has_five_seq:
        if bottom_points and convexHull == 3:
            classification = "self"
            start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
            return classification, start_point, end_point, approx
        
        elif convexHull and convexHull == 3:
            print("asynchronous")
            classification = "asynchronous"
            start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
            
            return classification, start_point, end_point, approx
        elif convexHull and convexHull <= 5:
            print("sychronous 374")
            classification = "synchronous"
            start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
            return classification, start_point, end_point, approx
        else: 
            print("Convex hull", convexHull)
            print("line")
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
        
        if (w > 15 or h > 15) and w+2 >= h or h > 50:  # filter small noise
            
            parent = hierarchy[0][i][3]
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
            
            if parent != -1:
                continue  # skip child contours
                
            count += 1
            # if(count != 58): continue
            
            classification, start_point, end_point, approx = classify_contour(
                contours, hierarchy, i
            )
            
            print(f"Contour {count}: {classification}, Start: {start_point}, End: {end_point}")
            x, y, w, h = cv2.boundingRect(approx)
            print("L58: ", len(approx))
            
            # get dashline
            if (w < 40 or ((classification == "asynchronous" or classification == "synchronous") and w < 100)):
                added = False
                for group in dashLineArr:
                    gx, gy, gw, gh = cv2.boundingRect(group[0])
                    
                    if abs(y - gy) < 20 : # same horizontal level
                        group.append(approx)
                        added = True
                        print("Y ", y, " ", gy , " count ", count)
                        break
                    
                  # If not added to any group, create a new one
                if not added:
                    print("newly Y ", y, count)
                    dashLineArr.append([cnt])

                continue  # Skip further processing for dash contours
            
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
    
    # print("58: ", contours_info)
    # print("Dash line groups details:", len(dashLineArr))
    c = 0
    
    # NEW: Process dash line groups to determine start/end for reply messages
    for group in dashLineArr:
        # Flatten all points from contours in the group
        all_points = np.vstack([cnt.reshape(-1, 2) for cnt in group])  # shape: (N, 2)

        # Calculate overall bounding box from all points
        x, y, w, h = cv2.boundingRect(all_points)
        group_min_x, group_min_y = x, y
        group_max_x, group_max_y = x + w, y + h

        # Sort group by X coordinate to ensure left-to-right order
        group_sorted = sorted(group, key=lambda cnt: cv2.boundingRect(cnt)[0])

        # Check first and last contours for arrowheads (>4 points)
        first_contour = group_sorted[0]
        last_contour = group_sorted[-1]

        first_approx_points = len(first_contour)
        last_approx_points = len(last_contour)

        print(f"Dash Group {c+1}: First contour points: {first_approx_points}, Last contour points: {last_approx_points}")

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
        x1, y1, x2, y2 = rect
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        
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

        # Loop through texts
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


def restructureData(rectangles, contours, texts):
    objectNodes = []
    messages = []
    notations = []

    # ids
    objectID = 0 
    messageID = 0
    notationID = 0
    text = ""
    print(len(texts))
  
    #detect the object 
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        w, h = x2 - x1, y2 - y1

        # Detect only object lifelines (based on your width rule)
        if w > h:
            label = extract_text_in_region(rect, texts, "obj")  # ✅ Reusable call
            text = (text + " " + label) if label else text
            objectNodes.append({
                "objId": objectID,
                "bbox": {"x": x1, "y": y1, "w": w, "h": h},
                "label": label  # None if no label found
            })
            objectID += 1
    
    for cont in contours:
        classification = cont["type"]
        startPoint = cont["start"]
        endPoint = cont["end"]
        # points = cont["approx"]
        text = ""
        if (classification == "asynchronous" or classification == "synchronous" or classification == "self" or classification == "reply"):
        
            text = extract_text_in_region(cont, texts, "message")
            messages.append({
                "messageID": messageID,
                "type": classification,
                "start": startPoint,
                "end": endPoint,
                "label": text,
            })
        
            messageID += 1
            
            print(classification, " : ", text)
            
        else :
            x, y, w, h = cv2.boundingRect(cont["approx"])
            text = extract_text_in_region(cont, texts, "notation")
            notations.append({
                "notationID": notationID,
                "type": classification,
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "label": text,
            })
            
            notationID += 1
            print(classification, " : ", text)
            
    print(len(texts))
    
    
    for msg in messages:
        start_point = msg["start"]
        end_point = msg["end"]

        # Check objects first
        sender = find_nearby_object(start_point, objectNodes)
        receiver = find_nearby_object(end_point, objectNodes)
        
        # If not found, check notations
        if sender is None:
            sender = find_nearby_notation(start_point, notations)
            
        if receiver is None:
            receiver = find_nearby_notation(end_point, notations)
        
        if sender is None and msg["type"] == "self":
                sender = receiver 
        
        print(msg["type"] )
        print("Start point:", start_point, "-> sender:", sender)
        print("End point:", end_point, "-> receiver:", receiver)
        print()
        
        
        msg["from"] = sender
        msg["to"] = receiver

    print("Messages: ", messages[5])
    
    
    # return {"objects": objectNodes, "remainingTexts": texts}


def find_nearby_object(point, objects, margin=5):
    px, py = point
    for obj in objects:
        x, y, w, h = obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["w"], obj["bbox"]["h"]
        # check if point is inside box with margin
        if (px >= x - margin) and (px <= x + w + margin) :
            return "obj" + str(obj["objId"])
    return None


def find_nearby_notation(point, notations, margin=5):
    px, py = point
    for note in notations:
        x, y, w, h = note["bbox"]["x"], note["bbox"]["y"], note["bbox"]["w"], note["bbox"]["h"]
        if (px >= x - margin) and (px <= x + w + margin):
            return ("notation"+str(note["notationID"]))
    return None


def merge_nearby_texts(texts, x_margin=10, y_margin=5):
    """
    Merge nearby text boxes into sentences.
    
    :param texts: list of tuples (x, y, w, h, text)
    :param x_margin: horizontal gap threshold to merge
    :param y_margin: vertical gap threshold to consider same line
    :return: merged list of tuples (x, y, w, h, sentence)
    """
    if not texts:
        return []

    # 1️⃣ Sort by top-left: top to bottom, then left to right
    texts_sorted = sorted(texts, key=lambda t: (t[1], t[0]))

    merged_texts = []
    current_group = [texts_sorted[0]]

    for entry in texts_sorted[1:]:
        x, y, w, h, text = entry
        last_x, last_y, last_w, last_h, last_text = current_group[-1]

        # Check if the current box is on the same line (vertical overlap)
        same_line = abs(y - last_y) <= y_margin

        # Check horizontal gap to previous box
        close_horizontally = (x - (last_x + last_w)) <= x_margin

        if same_line and close_horizontally:
            # Merge into current group
            current_group.append(entry)
        else:
            # Finalize previous group as one sentence
            x0 = min(e[0] for e in current_group)
            y0 = min(e[1] for e in current_group)
            x1 = max(e[0] + e[2] for e in current_group)
            y1 = max(e[1] + e[3] for e in current_group)
            sentence = " ".join(e[4] for e in current_group)
            merged_texts.append((x0, y0, x1 - x0, y1 - y0, sentence))
            # Start new group
            current_group = [entry]

    # Finalize last group
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
    
    # --- 3. OCR + bounding box info ---
    textData = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    d1 = pytesseract.image_to_string(textImg)
    # --- 4. Convert grayscale to BGR for drawing ---
    textImg = cv2.cvtColor(textImg, cv2.COLOR_GRAY2BGR)
    print("Contour ", len(contours))
    # --- 5. Draw bounding boxes ---
    n_boxes = len(textData['level'])
    for i in range(n_boxes):
        text = textData['text'][i].strip()
        conf = int(textData['conf'][i])

        if text != "":  # filter low confidence or empty
            (x, y, w, h) = (textData['left'][i], textData['top'][i], textData['width'][i], textData['height'][i])
            cv2.rectangle(textImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(textImg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # ✅ Convert pytesseract dict into clean list format
    texts = []
    for i in range(len(textData['text'])):
        text = textData['text'][i].strip()
        if text != "":  # ignore empty OCR outputs
            x = int(textData['left'][i])
            y = int(textData['top'][i])
            w = int(textData['width'][i])
            h = int(textData['height'][i])
            texts.append((x, y, w, h, text))
    
    # Merge nearby words into sentences
    texts_merged = merge_nearby_texts(texts)
    
    # restructure data
    restructureData(rectangles, contours, texts_merged)
    
    # visualize(thresh, drawContour, image2)
    visualize(image, drawContour, textImg)


# ---------------- Run ----------------
main("./images/sqd5.png")
