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

# ---------------- Modified: Draw rectangles & remove actual line areas ----------------
def draw_rectangles(image, rectangles, line_areas_to_remove, bg_color=(255, 255, 255)):
    image_rect = image.copy()
    
    # Remove the actual line areas that form the rectangles
    for line_area in line_areas_to_remove:
        x1, y1, x2, y2 = line_area
        # Fill the actual line area with background color
        image_rect[y1:y2, x1:x2] = bg_color
        
    return image_rect


# new code 9/27/2025 12AM
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
            print("Points: ", child_approx.reshape(-1, 2))
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
        
        if has_five_seq: #break if have 5-point sequence,
            print("meron")
            break
        # if the decimal is exceeding to the bsta pag lumagpas
        elif epsilon_ratio <= 0.005:
            print("greatdr", len(approx))
            break 
        
        epsilon_ratio = epsilon_ratio / 2  # decrease epsilon to get more detailed approx
    
    print("Classifying, Current Epsilon: ", epsilon_ratio)
    
    if has_five_seq:
        if convexHull and convexHull == 3:
            relationship_type = "directed association"
        else:
            relationship_type = "composition"
        # Determine start/end for 5-point sequence relationships
        start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
        print("Start", start_point, " End: ", end_point, " Point1: ", point1, " Point2: ", point2)
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

def find_five_point_sequence(points, min_tolerance=5, tolerance_percent=0.25):
    """Find and return the 5-point sequence if it exists."""
    n = len(points)
    if n < 5:
        return None, None
    # print(f"Approx Points: {points.reshape(-1, 2)}")
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
            print("Found 5 points - L267")
            print(f"Points: {seq_points}, Distances: {distances}")
            print(f"Hull Count: {hull_count}, Max: {max(distances):.2f}, Min: {min(distances):.2f}")
            # print(f"Dynamic Tolerance: {tolerance:.2f}")
            print("Tolerance1: ", max(distances)-min(distances))
            print("Tolerance: ",  dynamic_tolerance)
            if hull_count >= 4:
                return seq_points, hull_count
            elif hull_count == 3:
                return seq_points, hull_count
            
    print("No 5points L275")
    return None, None

# new code 10/1/2024 1AM - improved merging logic to handle chains and prevent arrow head merging
def detect_contours(image, rectangles, bg_color=0, min_area=100):
    detectedContours = image.copy()
    detectText = image.copy()
    
    kernel = np.ones((3, 3), np.uint8)
    detectedContours = cv2.dilate(detectedContours, kernel, iterations=1)  # fills small gaps
    detectedContours = cv2.erode(detectedContours, kernel, iterations=1)   # shrinks back, keeps merged parts

    # Find contours
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    ) 

    contour_img = cv2.cvtColor(detectedContours, cv2.COLOR_GRAY2BGR)
    contours_info = []
    count = 0

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        inTheRect = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            # Check if contour bbox is within rectangle bbox
            if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                # Fill rectangle area with background color to remove it
                inTheRect = True
                break  # No need to check other rectangles
            
        if inTheRect: continue  # Skip this contour entirely
        
        if w > 30 or h > 30:  # filter small noise
            parent = hierarchy[0][i][3]
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)  # Fill contour with background color
            
            if parent != -1:
                continue  # skip child contours
                
            count += 1
            # if(count != 6): continue
            classification, start_point, end_point, endpoint1, endpoint2, approx = classify_relationship(
                contours, hierarchy, i
            )

            print(f"Contour {count}: {classification}")
            print()
            # Draw bounding box
            cv2.rectangle(contour_img, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(contour_img, str(count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Draw approx points
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            for pt in approx:
                cv2.circle(contour_img, tuple(pt[0]), 1, (255, 0, 0), -1)

            # Save info for merging - ONLY CHANGE: added has_arrow_head flag
         
            contours_info.append({
                "classification": classification,
                "start": start_point,
                "end": end_point,
                "p1": endpoint1,
                "p2": endpoint2,
                "approx": approx,
                "has_arrow_head": classification in ["aggregation", "composition", "inheritance", "dependency", "directed association"]  # ONLY ADDITION
            })
            
            print("=========================================================")
          
    # 🔗 Merge broken contours
    merged_contours = connectContours(contours_info)
    
    return merged_contours, hierarchy, detectedContours, contour_img, detectText

 
def connectContours(contours_info, max_distance=30):
    """
    SIMPLE AND STRAIGHTFORWARD: Merge contours and use opposite endpoints
    """
    # Build adjacency graph
    graph = {i: [] for i in range(len(contours_info))}
    connection_info = {}  # Store which points connected between contours
    
    # Step 1: Find all possible connections
    for i in range(len(contours_info)):
        for j in range(i + 1, len(contours_info)):
            c1 = contours_info[i]
            c2 = contours_info[j]
            
            # ❌ Skip if both have arrow heads
            if c1["has_arrow_head"] and c2["has_arrow_head"]:
                continue
                
            points_c1 = [c1["p1"], c1["p2"]]
            points_c2 = [c2["p1"], c2["p2"]]
            
            # Check if contours should be connected
            connected_p1 = None
            connected_p2 = None
            should_connect = False
            
            for idx1, p1 in enumerate(points_c1):
                should_connect = False
                for idx2, p2 in enumerate(points_c2):
                    
                    if p1 is None or p2 is None:
                        continue
                    
                    if(c1["end"] is not None and p1 == c1["end"]) or (c2["end"] is not None and p2 == c2["end"]):
                        continue
                                       
                    dist = math.dist(p1, p2)
                    if dist < max_distance:
                        should_connect = True
                        connected_p1 = (idx1, p1)  # Store which endpoint from c1
                        connected_p2 = (idx2, p2)  # Store which endpoint from c2
                        print(f"🔗 Connection found: contour {i+1} point {p1} ↔ contour {j+1} point {p2} (distance={dist:.2f})")
                        break
                if should_connect:
                    break
            
            if should_connect:
                graph[i].append(j)
                graph[j].append(i)
                # Store connection info
                connection_info[(i, j)] = (connected_p1, connected_p2)
    
    # Step 2: Find connected components using DFS
    visited = set()
    connected_components = []
    
    def dfs(node, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                stack.extend(neighbor for neighbor in graph[current] if neighbor not in visited)
    
    for node in range(len(contours_info)):
        if node not in visited:
            component = []
            dfs(node, component)
            if len(component) > 1:  # Only merge if multiple contours
                # Check if component contains two arrow heads (should not merge)
                arrow_head_count = sum(1 for idx in component if contours_info[idx]["has_arrow_head"])
                if arrow_head_count < 2:  # Only merge if less than 2 arrow heads
                    connected_components.append(component)
    
    # Step 3: Merge contours and find chain endpoints
    merged_contours_info = []
    used_indices = set()
    
    for component in connected_components:
        print(f"🔄 Merging component: {[idx+1 for idx in component]}")
        
        # Merge all contour points
        all_points = []
        component_classifications = []
        for idx in component:
            all_points.append(contours_info[idx]["approx"])
            component_classifications.append(contours_info[idx]["classification"])
            used_indices.add(idx)
        
        merged = np.vstack(all_points)
        epsilon = 0.01 * cv2.arcLength(merged, True)
        merged_approx = cv2.approxPolyDP(merged, epsilon, True)
        
        # 🔥 STRAIGHTFORWARD: Find the two opposite endpoints of the chain
        endpoint1, endpoint2 = find_chain_endpoints_simple(component, connection_info, contours_info)
        
        # 🔥 NEW: Determine the classification for merged contour
        merged_classification = determine_merged_classification(component_classifications)
        
        # Check if merged contour should have arrow head
        has_arrow_head = merged_classification in ["aggregation", "composition", "inheritance", "dependency", "directed association"]
        
        # 🔥 NEW: Properly assign start and end points
        start_point, end_point = determine_start_end_points(component, contours_info, endpoint1, endpoint2, has_arrow_head)
        
        # Store merged contour
        merged_contours_info.append({
            "approx": merged_approx,
            "p1": endpoint1,
            "p2": endpoint2,
            "start": start_point,  # Properly assigned start point
            "end": end_point,      # Properly assigned end point
            "classification": merged_classification,
            "has_arrow_head": has_arrow_head
        })
        
        print(f"📏 Merged contour endpoints: {endpoint1} ↔ {endpoint2}")
        print(f"📍 Start point: {start_point}, End point: {end_point}")
        print(f"🏷️  Merged classification: {merged_classification}")
    
    # Step 4: Add all non-merged contours
    new_contours = []
    for idx, c in enumerate(contours_info):
        if idx not in used_indices:
            new_contours.append(c)
    
    new_contours.extend(merged_contours_info)
    
    print(f"📊 Contour merging summary: {len(contours_info)} original → {len(new_contours)} final contours")
    
    return new_contours


def determine_merged_classification(classifications):
    """
    Determine the classification for merged contours based on priority:
    1. If any contour has a special relationship (non-association), use that
    2. If all are associations, use association
    3. Default to association
    """
    # Define relationship priorities (non-association relationships have priority)
    priority_relationships = ["inheritance", "composition", "aggregation", "dependency", "directed association"]
    
    for relationship in priority_relationships:
        if relationship in classifications:
            return relationship
    
    # If no special relationships found, check for association
    if "association" in classifications:
        return "association"
    
    # Default fallback
    return "association"


def find_chain_endpoints_simple(component, connection_info, contours_info):
    """
    SIMPLE: Find the two endpoints that were never connected to anything
    """
    if len(component) == 1:
        # Single contour - just return its original endpoints
        idx = component[0]
        return contours_info[idx]["p1"], contours_info[idx]["p2"]
    
    # Track which endpoints were used for connections
    used_endpoints = set()
    
    # Mark all endpoints that were used for connections
    for i in range(len(component)):
        for j in range(i + 1, len(component)):
            idx1, idx2 = component[i], component[j]
            if (idx1, idx2) in connection_info:
                conn1, conn2 = connection_info[(idx1, idx2)]
                used_endpoints.add((idx1, conn1[0]))  # (contour_index, endpoint_index)
                used_endpoints.add((idx2, conn2[0]))
            elif (idx2, idx1) in connection_info:
                conn2, conn1 = connection_info[(idx2, idx1)]
                used_endpoints.add((idx1, conn1[0]))
                used_endpoints.add((idx2, conn2[0]))
    
    # Find endpoints that were NEVER used for connections - these are our chain endpoints
    chain_endpoints = []
    
    for idx in component:
        contour = contours_info[idx]
        # Check both endpoints (0 = p1, 1 = p2)
        for endpoint_idx, endpoint in enumerate([contour["p1"], contour["p2"]]):
            if endpoint is not None and (idx, endpoint_idx) not in used_endpoints:
                chain_endpoints.append(endpoint)
    
    # Return the two endpoints (or whatever we found)
    if len(chain_endpoints) >= 2:
        return chain_endpoints[0], chain_endpoints[1]
    elif len(chain_endpoints) == 1:
        return chain_endpoints[0], None
    else:
        # Fallback: just use first and last contour's endpoints
        first_contour = contours_info[component[0]]
        last_contour = contours_info[component[-1]]
        return first_contour["p1"], last_contour["p2"]


def determine_start_end_points(component, contours_info, endpoint1, endpoint2, has_arrow_head):
    """
    Determine proper start and end points for merged contour
    """
    start_point = None
    end_point = None
    
    if not has_arrow_head:
        # For associations without arrow heads, no specific start/end
        return None, None
    
    # Find which original contour had the arrow head
    arrow_head_contour = None
    for idx in component:
        if contours_info[idx]["end"] is not None:
            arrow_head_contour = contours_info[idx]
            break
    
    if arrow_head_contour and arrow_head_contour["end"] is not None:
        arrow_head_location = arrow_head_contour["end"]
        
        # Determine which merged endpoint is closer to the original arrow head location
        dist_to_endpoint1 = math.dist(endpoint1, arrow_head_location) if endpoint1 else float('inf')
        dist_to_endpoint2 = math.dist(endpoint2, arrow_head_location) if endpoint2 else float('inf')
        
        if dist_to_endpoint1 < dist_to_endpoint2:
            # endpoint1 is the arrow head (end point)
            end_point = endpoint1
            start_point = endpoint2
        else:
            # endpoint2 is the arrow head (end point)
            end_point = endpoint2
            start_point = endpoint1
    else:
        # Fallback: if we can't find the original arrow head, use endpoint ordering
        start_point = endpoint1
        end_point = endpoint2
    
    return start_point, end_point


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


def detect_separator_lines_in_rectangle(rectangle, image):
    """Detect vertical separator lines within a rectangle"""
    x1, y1, x2, y2 = rectangle
    
    # Crop the rectangle area from the image
    rect_roi = image[y1:y2, x1:x2]
    
    # Convert to grayscale and threshold
    gray_roi = cv2.cvtColor(rect_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Detect vertical lines within the rectangle
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_lines = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Find contours of vertical lines
    line_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and get vertical separator lines (long vertical lines)
    separator_lines = []
    for cnt in line_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Look for vertical lines that are tall enough (at least 30% of rectangle height)
        if h > (y2 - y1) * 0.3 and w < 10:  # Vertical lines are thin and tall
            # Convert local coordinates to global coordinates
            global_y = y1 + y
            separator_lines.append(global_y)
    
    # We expect exactly 2 separator lines, if we find more, take the two with highest Y values
    if len(separator_lines) >= 2:
        separator_lines.sort()
        # Take the two lines that are most likely to be the separators
        line1 = separator_lines[0]
        line2 = separator_lines[1]
        return line1, line2
    else:
        # If we can't detect lines, use heuristic based on rectangle height
        rect_height = y2 - y1
        line1 = y1 + rect_height * 0.2  # 20% from top
        line2 = y1 + rect_height * 0.6  # 60% from top
        return line1, line2

def separate_access_modifier(text):
    """Separate access modifier from the rest of the text"""
    access_modifiers = ['+', '-', '#', '~']  # public, private, protected, package
    
    for modifier in access_modifiers:
        if text.startswith(modifier):
            # Return modifier and the rest of the text (without the modifier)
            return modifier, text[len(modifier):].strip()
    
    # If no access modifier found, assume public (+) as default
    return '+', text

def process_text_lines_with_modifiers(text_items):
    """Group text items by line and separate access modifiers"""
    if not text_items:
        return []
    
    # Sort by Y then X
    text_items.sort(key=lambda item: (item['y'], item['x']))
    
    lines = []
    current_line = []
    current_line_y = None
    line_height_threshold = 20
    
    for item in text_items:
        y_pos = item['y']
        
        if current_line_y is None or abs(y_pos - current_line_y) <= line_height_threshold:
            if current_line_y is None:
                current_line_y = y_pos
            current_line.append(item)
        else:
            # Process completed line
            if current_line:
                current_line.sort(key=lambda item: item['x'])
                line_text = ' '.join(item['text'] for item in current_line)
                modifier, content = separate_access_modifier(line_text)
                lines.append([modifier, content])
            current_line = [item]
            current_line_y = y_pos
    
    # Process last line
    if current_line:
        current_line.sort(key=lambda item: item['x'])
        line_text = ' '.join(item['text'] for item in current_line)
        modifier, content = separate_access_modifier(line_text)
        lines.append([modifier, content])
    
    return lines

def find_closest_class(classes, x, y):
    """Find the class whose rectangle is closest to the given point"""
    closest_class = None
    min_distance = float('inf')
    
    for class_info in classes:
        rect_x1, rect_y1, rect_x2, rect_y2 = class_info['position']
        
        # Calculate distance from point to rectangle
        dx = max(rect_x1 - x, 0, x - rect_x2)
        dy = max(rect_y1 - y, 0, y - rect_y2)
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < min_distance:
            min_distance = distance
            closest_class = class_info['name']
    
    return closest_class

def normalize_relationship_type(relationship_type):
    """Normalize relationship type names"""
    type_mapping = {
        'inheritance': 'inheritance',
        'composition': 'composition', 
        'aggregation': 'aggregation',
        'directed association': 'association',
        'association (plain line)': 'association',
        'association': 'association',
        'dependency': 'dependency'
    }
    return type_mapping.get(relationship_type, 'association')

def process_multiplicities(outside_texts, contours, rectangles):
    """Process outside text as multiplicities"""
    multiplicities = []
    
    # Common multiplicity patterns
    multiplicity_patterns = ['0..1', '1..*', '0..*', '1', '*', '0..n', '1..n', '..']
    
    for text_item in outside_texts:
        text = text_item['text']
        x, y = text_item['x'], text_item['y']
        
        # Check if text matches common multiplicity patterns
        is_multiplicity = any(pattern in text for pattern in multiplicity_patterns)
        
        if is_multiplicity:
            # Find which relationship this multiplicity belongs to
            closest_relationship = find_closest_relationship(contours, x, y)
            closest_class = find_closest_class_from_rectangles(rectangles, x, y)
            
            multiplicities.append({
                'text': text,
                'position': (x, y),
                'relationship': closest_relationship,
                'class': closest_class
            })
    
    return multiplicities

def find_closest_relationship(contours, x, y):
    """Find the relationship contour closest to the given point"""
    closest_relationship = None
    min_distance = float('inf')
    
    for i, cont in enumerate(contours):
        if cont['start'] and cont['end']:
            # Calculate distance to line segment midpoint
            start_x, start_y = cont['start']
            end_x, end_y = cont['end']
            
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            distance = math.sqrt((x - mid_x)**2 + (y - mid_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_relationship = i
    
    return closest_relationship

def find_closest_class_from_rectangles(rectangles, x, y):
    """Find the class rectangle closest to the given point"""
    closest_class = None
    min_distance = float('inf')
    
    for i, rect in enumerate(rectangles):
        rx1, ry1, rx2, ry2 = rect
        # Calculate distance to rectangle center
        rect_center_x = (rx1 + rx2) / 2
        rect_center_y = (ry1 + ry2) / 2
        distance = math.sqrt((x - rect_center_x)**2 + (y - rect_center_y)**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_class = i
    
    return closest_class

def find_multiplicity_for_endpoint(multiplicities, relationship_index, contour, endpoint):
    """Find multiplicity for a specific relationship endpoint"""
    if endpoint == 'start' and not contour['start']:
        return None
    if endpoint == 'end' and not contour['end']:
        return None
        
    target_point = contour['start'] if endpoint == 'start' else contour['end']
    
    closest_multiplicity = None
    min_distance = float('inf')
    
    for multiplicity in multiplicities:
        if multiplicity['relationship'] == relationship_index:
            mult_x, mult_y = multiplicity['position']
            distance = math.dist((mult_x, mult_y), target_point)
            
            if distance < min_distance:
                min_distance = distance
                closest_multiplicity = multiplicity['text']
    
    return closest_multiplicity

def process_relationships_with_multiplicities(contours, classes, multiplicities):
    """Process relationships and attach multiplicities directly"""
    relationships = []
    
    for i, cont in enumerate(contours):
        relationship_type = cont['classification'].lower()
        
        # Skip if it's just an association without specific type
        if 'association' in relationship_type and relationship_type == 'association':
            continue
            
        # Find connected classes
        start_class = None
        end_class = None
        
        if cont['start'] and cont['end']:
            start_x, start_y = cont['start']
            end_x, end_y = cont['end']
            
            start_class = find_closest_class(classes, start_x, start_y)
            end_class = find_closest_class(classes, end_x, end_y)
        
        # If start/end not available, use p1 and p2 as fallback
        if not start_class and cont['p1']:
            start_x, start_y = cont['p1']
            start_class = find_closest_class(classes, start_x, start_y)
            
        if not end_class and cont['p2']:
            end_x, end_y = cont['p2']
            end_class = find_closest_class(classes, end_x, end_y)
        
        if start_class and end_class and start_class != end_class:
            # Find multiplicities for this relationship
            start_multiplicity = find_multiplicity_for_endpoint(multiplicities, i, cont, 'start')
            end_multiplicity = find_multiplicity_for_endpoint(multiplicities, i, cont, 'end')
            
            relationship_info = {
                'from_class': start_class,
                'to_class': end_class,
                'relationship_type': normalize_relationship_type(relationship_type),
                'multiplicities': {
                    'from_end': start_multiplicity,
                    'to_end': end_multiplicity
                }
            }
            relationships.append(relationship_info)
    
    return relationships

def reStructureData(rectangles, contours, ocrData, original_image):
    structuredData = {
        'classes': [],
        'relationships': []  # Multiplicities are stored directly in relationships
    }
    
    # Separate text into inside and outside class rectangles
    inside_texts = []
    outside_texts = []
    
    n_boxes = len(ocrData['level'])
    for j in range(n_boxes):
        text = ocrData['text'][j].strip()
        x = ocrData['left'][j]
        y = ocrData['top'][j]
        w = ocrData['width'][j]
        h = ocrData['height'][j]
        
        if not text:
            continue
            
        # Check if text is inside ANY rectangle
        text_inside_any_rect = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            if (x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2):
                text_inside_any_rect = True
                break
        
        if text_inside_any_rect:
            inside_texts.append({
                'text': text, 'x': x, 'y': y, 'w': w, 'h': h
            })
        else:
            outside_texts.append({
                'text': text, 'x': x, 'y': y, 'w': w, 'h': h
            })
    
    print(f"Found {len(inside_texts)} text elements inside classes")
    print(f"Found {len(outside_texts)} text elements outside classes")
    
    # Process rectangles as classes (using only inside texts)
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect
        
        # Detect separator lines for this rectangle
        line1_y, line2_y = detect_separator_lines_in_rectangle(rect, original_image)
        print(f"Class {i+1} separators: Line1 at {line1_y}, Line2 at {line2_y}")
        
        # Find text within THIS specific rectangle area
        class_name_texts = []
        attribute_texts = []
        method_texts = []
        
        for text_item in inside_texts:
            x = text_item['x']
            y = text_item['y']
            w = text_item['w']
            h = text_item['h']
            text = text_item['text']
            
            # Check if text is inside this specific rectangle
            if (x >= x1 and y >= y1 and (x + w) <= x2 and (y + h) <= y2):
                
                # Calculate text center Y position
                text_center_y = y + h/2
                
                # Classify text based on separator lines
                if text_center_y < line1_y:
                    # Above first line - Class name
                    class_name_texts.append({
                        'y': y, 'text': text, 'x': x
                    })
                elif line1_y <= text_center_y <= line2_y:
                    # Between two lines - Attributes
                    attribute_texts.append({
                        'y': y, 'text': text, 'x': x
                    })
                else:
                    # Below second line - Methods
                    method_texts.append({
                        'y': y, 'text': text, 'x': x
                    })
        
        # Process class name
        class_name = f"Class{i+1}"
        if class_name_texts:
            class_name_texts.sort(key=lambda item: (item['y'], item['x']))
            class_name = ' '.join(item['text'] for item in class_name_texts)
        
        # Process attributes and methods with access modifiers
        attributes = process_text_lines_with_modifiers(attribute_texts)
        methods = process_text_lines_with_modifiers(method_texts)
        
        class_info = {
            'name': class_name,
            'attributes': attributes,
            'methods': methods,
            'position': (x1, y1, x2, y2)
        }
        structuredData['classes'].append(class_info)
    
    # Process outside texts as multiplicities
    multiplicities = process_multiplicities(outside_texts, contours, rectangles)
    
    # Process relationships with multiplicities attached
    structuredData['relationships'] = process_relationships_with_multiplicities(
        contours, structuredData['classes'], multiplicities
    )
    
    return structuredData

def print_structured_data(structuredData):
    """Print the complete structured data in a readable format"""
    print("\n" + "="*80)
    print("COMPLETE UML STRUCTURED DATA")
    print("="*80)
    
    print("\nCLASSES:")
    print("-" * 40)
    for i, class_info in enumerate(structuredData['classes']):
        print(f"\n{i+1}. {class_info['name']} {class_info['position']}")
        
        print("   Attributes:")
        for attr in class_info['attributes']:
            print(f"     [{attr[0]}] {attr[1]}")
            
        print("   Methods:")
        for method in class_info['methods']:
            print(f"     [{method[0]}] {method[1]}")
    
    print("\nRELATIONSHIPS:")
    print("-" * 40)
    for i, rel in enumerate(structuredData['relationships']):
        from_mult = rel['multiplicities']['from_end'] or '1'
        to_mult = rel['multiplicities']['to_end'] or '1'
        
        print(f"\n{i+1}. {rel['from_class']}[{from_mult}] -- {rel['relationship_type']} --> {rel['to_class']}[{to_mult}]")
        
        # Relationship explanations
        explanations = {
            'inheritance': f"    {rel['to_class']} is parent of {rel['from_class']}",
            'composition': f"    {rel['from_class']} contains {rel['to_class']} (strong ownership)",
            'aggregation': f"    {rel['from_class']} uses {rel['to_class']} (weak ownership)",
            'association': f"    {rel['from_class']} is associated with {rel['to_class']}",
            'dependency': f"    {rel['from_class']} depends on {rel['to_class']}"
        }
        print(explanations.get(rel['relationship_type'], "    Unknown relationship type"))


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
    image_rect = draw_rectangles(image, rectangles, lines_to_remove)
    
    # ==================== Preprocess again the image that after removing teh rect =============================== 
    noRectImg = image_rect.copy()      #copy image after removing rectangles
    threshImg = threshold_image(cv2.cvtColor(noRectImg, cv2.COLOR_BGR2GRAY))    #threshold again
    
    contours, hierarchy, detectedContours, drawContour, textImg = detect_contours(threshImg, rectangles)
    # downloadImage(textImg)
    image2 = image.copy()
    # for i, cont in enumerate(contours):
    #     print(f"Contour {i+1}: {cont['classification']}, Start: {cont['start']}, End: {cont['end']}, P1: {cont['p1']}, P2: {cont['p2']}")
    #     # print("Approx Points: ", cont['approx'].reshape(-1, 2))
    #     # Draw contours
    #     cv2.drawContours(image2, [cont['approx']], -1, (255, 0, 255), 1)
        
    #     # Draw start/end points if they exist
    #     if cont['start']:
    #         cv2.circle(image2, cont['start'], 4, (0, 255, 0), -1)  # Green for start
    #     if cont['end']:
    #         cv2.circle(image2, cont['end'], 4, (0, 0, 255), -1)    # Red for end
        
    #     # Draw bounding rectangle
    #     x, y, w, h = cv2.boundingRect(cont['approx'])
    #     cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 255), 1)  # Yellow rectangle
        
    #     # Optional: Add contour number on bounding box
    #     cv2.putText(image2, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
    #                 0.5, (0, 255, 0), 1)
    #     for point in cont['approx']:
    #         cv2.circle(image2, tuple(point[0]), 1, (255, 0, 0), -1)  # Blue for approx points
    
    
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
    
    structuredData = reStructureData(rectangles, contours, data, textImg)
    
    # Print the complete structured data
    print_structured_data(structuredData)
   
    visualize(textImg, drawContour, image2)


# ---------------- Run ----------------
main("./images/cd_comp.png")
