import cv2
import numpy as np
import math
import pytesseract
from pytesseract import Output

# Global parameter used for polygon approximation
epsilon_ratio = 0.04

# ---------------- Step 0: Load image ----------------
def load_image(path):
    """
    Load image from disk and return original BGR image and a grayscale copy.
    """
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# ---------------- Step 1: Threshold ----------------
def threshold_image(gray, thresh_val=128):
    """
    Binary inverse threshold the grayscale image.
    """
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return thresh

# ---------------- Step 2: Morphology ----------------
def extract_lines(thresh, vert_len=15, horiz_len=15):
    """
    Extract long vertical and horizontal lines using morphological opening.
    Returns two binary images containing vertical_lines and horizontal_lines.
    """
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))

    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    return vertical_lines, horizontal_lines

# ---------------- Step 3: Contour detection ----------------
def get_segments(lines):
    """
    Find external contours on a binary image and return bounding rects (x, y, w, h).
    """
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = [cv2.boundingRect(c) for c in contours]  # (x, y, w, h)
    return segments

# ---------------- Step 4: Find aligned pairs ----------------
def find_vertical_pairs(segments, y_tol=5, h_tol=5):
    """
    Among vertical segments (bounding boxes), find pairs aligned in Y and similar height.
    """
    pairs = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(y1 - y2) <= y_tol and abs(h1 - h2) <= h_tol:
                pairs.append((segments[i], segments[j]))
    return pairs

def find_horizontal_pairs(segments, x_tol=5, w_tol=5):
    """
    Among horizontal segments (bounding boxes), find pairs aligned in X and similar width.
    """
    pairs = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            x1, y1, w1, h1 = segments[i]
            x2, y2, w2, h2 = segments[j]
            if abs(x1 - x2) <= x_tol and abs(w1 - w2) <= w_tol:
                pairs.append((segments[i], segments[j]))
    return pairs

# ---------------- Step 5: Detect rectangles ----------------
def detect_rectangles(vertical_pairs, horizontal_pairs, vertical_lines, horizontal_lines):
    """
    Attempt to detect rectangle regions formed by pairs of vertical and horizontal line segments.
    Returns:
      - rectangles: list of (x1, y1, x2, y2)
      - line_areas_to_remove: list of bounding areas for the lines that form those rectangles
    Note: vertical_lines/horizontal_lines are accepted for compatibility but not used directly.
    """
    rectangles = []
    line_areas_to_remove = []

    for verticalpair in vertical_pairs:
        v1, v2 = verticalpair
        v_left_x = min(v1[0], v2[0])
        v_right_x = max(v1[0], v2[0]) + max(v1[2], v2[2])
        v_top_y = min(v1[1], v2[1])
        v_bottom_y = max(v1[1] + v1[3], v2[1] + v2[3])

        for horizontalpair in horizontal_pairs:
            h1, h2 = horizontalpair
            h_top_y = min(h1[1], h2[1])
            h_bottom_y = max(h1[1] + h1[3], h2[1] + h2[3])
            h_left_x = min(h1[0], h2[0])
            h_right_x = max(h1[0] + h1[2], h2[0] + h2[2])

            # Check containment: horizontal pair fully inside vertical pair bounds and vice-versa
            if h_left_x >= v_left_x and h_right_x <= v_right_x and v_top_y >= h_top_y and v_bottom_y <= h_bottom_y:
                h1Width = max(h1[2], h2[2])
                maxWidth = v_right_x - v_left_x
                v1Height = max(v1[3], v2[3])
                maxHeight = h_bottom_y - h_top_y

                # Heuristic tolerance to ensure the detected rectangle is consistent
                if ((maxWidth - h1Width) < 40 and (maxHeight - v1Height) < 40):
                    rectangles.append((v_left_x, h_top_y, v_right_x, h_bottom_y))

                    # Record precise line areas that form this rectangle to remove them later
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
    """
    Return a copy of image where the line areas that formed detected rectangles are overwritten
    with the background color (bg_color). Does not change original image.
    """
    image_rect = image.copy()
    for line_area in line_areas_to_remove:
        x1, y1, x2, y2 = line_area
        image_rect[y1:y2, x1:x2] = bg_color
    return image_rect

# ---------------- Relationship classification helpers ----------------
def classify_relationship(contours, hierarchy, idx):
    """
    Classify a contour (line + attached child symbol) into UML relationship type.
    Returns (relationship_type, start_point, end_point, endpoint1, endpoint2, approx_points)
    - For plain association or when endpoints can't be determined, start/end may be None.
    """
    global epsilon_ratio
    cnt = contours[idx]
    child_idx = hierarchy[0][idx][2]

    # reset epsilon_ratio locally to original default for deterministic approximation loop
    epsilon_ratio = 0.04
    approx = None

    while True:
        epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Determine farthest endpoints on this approx
        farthest_points = find_farthest_points(approx)
        if farthest_points is None:
            # Fallback: treat as plain association with returned approx
            return "Association (plain line)", None, None, None, None, approx

        point1, point2, max_distance = farthest_points

        # If this contour has a child in hierarchy, try to classify child (arrowhead/diamond)
        if child_idx != -1:
            child_cnt = contours[child_idx]
            child_epsilon = 0.08 * cv2.arcLength(child_cnt, True)
            child_approx = cv2.approxPolyDP(child_cnt, child_epsilon, True)
            child_vertices = len(child_approx)

            # Classify common UML symbols by vertex count
            if child_vertices == 3:
                relationship_type = "inheritance"
            elif child_vertices == 4:
                relationship_type = "aggregation"
            else:
                relationship_type = "Other unfilled shape"

            # Determine direction: which farthest point is closer to the child symbol
            start_point, end_point = determine_start_end_with_child(point1, point2, child_approx)

            return relationship_type, start_point, end_point, point1, point2, approx

        # Priority check: detect 5-point sequence (filled diamond/composition)
        has_five_seq, convexHull = find_five_point_sequence(approx)

        if has_five_seq:
            break
        elif epsilon_ratio <= 0.005:
            break

        # Make polygon approximation more detailed and retry
        epsilon_ratio = epsilon_ratio / 2

    # After loop: decide based on whether 5-point sequence existed
    if has_five_seq:
        if convexHull and convexHull == 3:
            relationship_type = "directed association"
        else:
            relationship_type = "composition"
        start_point, end_point = determine_start_end_five_sequence(point1, point2, approx)
        return relationship_type, start_point, end_point, point1, point2, approx
    else:
        relationship_type = "association"
        return relationship_type, None, None, point1, point2, approx

def find_farthest_points(approx_points):
    """
    From approx points (N,1,2) find two points with maximum Euclidean distance.
    Returns (point1, point2, max_distance) as tuples or None if insufficient points.
    """
    points = np.squeeze(approx_points)  # (N,2) if N>1

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
    """
    Given two endpoints and a child symbol polygon, return (start_point, end_point)
    where end_point is the endpoint closer to the child symbol.
    """
    child_points = np.squeeze(child_approx)

    if len(child_points) == 0:
        return point1, point2

    dist1 = min([np.linalg.norm(np.array(point1) - child_point) for child_point in child_points])
    dist2 = min([np.linalg.norm(np.array(point2) - child_point) for child_point in child_points])

    # The point closer to the child is considered the 'end'
    if dist1 < dist2:
        return point2, point1
    else:
        return point1, point2

def determine_start_end_five_sequence(point1, point2, approx):
    """
    For a detected 5-point sequence, determine which of the farthest endpoints
    belongs to the 5-point sequence (that endpoint is the 'end').
    """
    points = np.squeeze(approx)
    five_seq_points, _ = find_five_point_sequence(points)

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

def find_five_point_sequence(points, min_tolerance=5, tolerance_percent=0.25):
    """
    Try to find a subsequence of 5 consecutive points (circularly) that form
    roughly equal edge lengths and produce a convex hull with sufficient vertices.
    Returns (seq_points, hull_count) or (None, None).
    """
    n = len(points)
    if n < 5:
        return None, None

    for start in range(n + (n - 1)):
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

        # Ratio scaling heuristic to compute a dynamic tolerance
        original_max = 30
        original_min = 18
        scale_factor = max_dist / original_max
        scaled_min = original_min * scale_factor

        dynamic_tolerance = max_dist - scaled_min

        if max(distances) - min(distances) <= dynamic_tolerance + 5:
            seq_np = np.array(seq_points, dtype=np.int32)
            hull = cv2.convexHull(seq_np, returnPoints=True)
            hull_count = len(hull)

            if hull_count >= 4:
                return seq_points, hull_count
            elif hull_count == 3:
                return seq_points, hull_count

    return None, None

# ---------------- New merging & contour detection code ----------------
def detect_contours(image, rectangles, bg_color=0, min_area=100):
    """
    Find contours in the (binary) image after removing rectangles.
    Returns merged_contours (output of connectContours), hierarchy, detectedContours (processed binary image),
    contour_img (visualization image), detectText (image with text contours filled).
    """
    detectedContours = image.copy()
    detectText = image.copy()

    # Small morphological operations to fill and re-shrink gaps
    kernel = np.ones((3, 3), np.uint8)
    detectedContours = cv2.dilate(detectedContours, kernel, iterations=1)
    detectedContours = cv2.erode(detectedContours, kernel, iterations=1)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        detectedContours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    contour_img = cv2.cvtColor(detectedContours, cv2.COLOR_GRAY2BGR)
    contours_info = []
    count = 0

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip contours that lie entirely inside any detected rectangle region
        inTheRect = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            if x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2:
                inTheRect = True
                break
        if inTheRect:
            continue

        # Filter by reasonably large bounding boxes (small noise ignored)
        if w > 30 or h > 30:
            parent = hierarchy[0][i][3]
            # Fill contour on detectText to avoid OCR picking up line pixels
            cv2.drawContours(detectText, [cnt], -1, bg_color, -1)

            # Skip any child contour in hierarchy (we process parents only)
            if parent != -1:
                continue

            count += 1

            classification, start_point, end_point, endpoint1, endpoint2, approx = classify_relationship(
                contours, hierarchy, i
            )

            # Visualization: bounding box and count label (kept for debugging/inspection image)
            cv2.rectangle(contour_img, (x - 1, y - 1), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(contour_img, str(count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Approx points (plotted on contour_img)
            epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            for pt in approx:
                cv2.circle(contour_img, tuple(pt[0]), 1, (255, 0, 0), -1)

            # Save contour info for later merging
            contours_info.append({
                "classification": classification,
                "start": start_point,
                "end": end_point,
                "p1": endpoint1,
                "p2": endpoint2,
                "approx": approx,
                "has_arrow_head": classification in ["aggregation", "composition", "inheritance", "dependency", "directed association"]
            })

    # Merge broken contour chains based on endpoints and heuristics
    merged_contours = connectContours(contours_info)

    return merged_contours, hierarchy, detectedContours, contour_img, detectText

def connectContours(contours_info, max_distance=30):
    """
    Merge nearby contour segments into chains using endpoint proximity.
    1) Build adjacency graph of connectable contours
    2) Extract connected components (chains)
    3) Merge contours within a component and compute chain endpoints
    4) Preserve non-merged contours
    """
    graph = {i: [] for i in range(len(contours_info))}
    connection_info = {}

    # Build possible connections between contour endpoints
    for i in range(len(contours_info)):
        for j in range(i + 1, len(contours_info)):
            c1 = contours_info[i]
            c2 = contours_info[j]

            # Do not connect two contours if both have arrow heads (avoid merging arrow symbols)
            if c1["has_arrow_head"] and c2["has_arrow_head"]:
                continue

            points_c1 = [c1["p1"], c1["p2"]]
            points_c2 = [c2["p1"], c2["p2"]]

            connected_p1 = None
            connected_p2 = None
            should_connect = False

            for idx1, p1 in enumerate(points_c1):
                should_connect = False
                for idx2, p2 in enumerate(points_c2):

                    if p1 is None or p2 is None:
                        continue

                    # Avoid connecting if endpoint equals an arrow-head 'end' endpoint of that contour
                    if (c1["end"] is not None and p1 == c1["end"]) or (c2["end"] is not None and p2 == c2["end"]):
                        continue

                    dist = math.dist(p1, p2)
                    if dist < max_distance:
                        should_connect = True
                        connected_p1 = (idx1, p1)
                        connected_p2 = (idx2, p2)
                        break
                if should_connect:
                    break

            if should_connect:
                graph[i].append(j)
                graph[j].append(i)
                connection_info[(i, j)] = (connected_p1, connected_p2)

    # Find connected components via DFS
    visited = set()
    connected_components = []

    def dfs(node, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)

    for node in range(len(contours_info)):
        if node not in visited:
            component = []
            dfs(node, component)
            if len(component) > 1:
                # Do not merge a component that contains two or more arrow heads
                arrow_head_count = sum(1 for idx in component if contours_info[idx]["has_arrow_head"])
                if arrow_head_count < 2:
                    connected_components.append(component)

    # Merge contours within each connected component
    merged_contours_info = []
    used_indices = set()

    for component in connected_components:
        all_points = []
        component_classifications = []
        for idx in component:
            all_points.append(contours_info[idx]["approx"])
            component_classifications.append(contours_info[idx]["classification"])
            used_indices.add(idx)

        # Stack all approx points into one array and approximate merged polygon
        merged = np.vstack(all_points)
        epsilon = 0.01 * cv2.arcLength(merged, True)
        merged_approx = cv2.approxPolyDP(merged, epsilon, True)

        # Find chain endpoints (the two endpoints never used for connections)
        endpoint1, endpoint2 = find_chain_endpoints_simple(component, connection_info, contours_info)

        # Determine merged classification using priority rules
        merged_classification = determine_merged_classification(component_classifications)

        # Decide whether merged chain should have arrow head
        has_arrow_head = merged_classification in ["aggregation", "composition", "inheritance", "dependency", "directed association"]

        # Assign start and end points for merged contour
        start_point, end_point = determine_start_end_points(component, contours_info, endpoint1, endpoint2, has_arrow_head)

        merged_contours_info.append({
            "approx": merged_approx,
            "p1": endpoint1,
            "p2": endpoint2,
            "start": start_point,
            "end": end_point,
            "classification": merged_classification,
            "has_arrow_head": has_arrow_head
        })

    # Add non-merged contours unchanged
    new_contours = []
    for idx, c in enumerate(contours_info):
        if idx not in used_indices:
            new_contours.append(c)

    new_contours.extend(merged_contours_info)
    return new_contours

def determine_merged_classification(classifications):
    """
    Choose a relationship classification for a merged component based on priority.
    """
    priority_relationships = ["inheritance", "composition", "aggregation", "dependency", "directed association"]

    for relationship in priority_relationships:
        if relationship in classifications:
            return relationship

    if "association" in classifications:
        return "association"

    return "association"

def find_chain_endpoints_simple(component, connection_info, contours_info):
    """
    Simple heuristic to find the chain endpoints:
    endpoints that were never used in any connection are the chain ends.
    """
    if len(component) == 1:
        idx = component[0]
        return contours_info[idx]["p1"], contours_info[idx]["p2"]

    used_endpoints = set()

    for i in range(len(component)):
        for j in range(i + 1, len(component)):
            idx1, idx2 = component[i], component[j]
            if (idx1, idx2) in connection_info:
                conn1, conn2 = connection_info[(idx1, idx2)]
                used_endpoints.add((idx1, conn1[0]))
                used_endpoints.add((idx2, conn2[0]))
            elif (idx2, idx1) in connection_info:
                conn2, conn1 = connection_info[(idx2, idx1)]
                used_endpoints.add((idx1, conn1[0]))
                used_endpoints.add((idx2, conn2[0]))

    chain_endpoints = []
    for idx in component:
        contour = contours_info[idx]
        for endpoint_idx, endpoint in enumerate([contour["p1"], contour["p2"]]):
            if endpoint is not None and (idx, endpoint_idx) not in used_endpoints:
                chain_endpoints.append(endpoint)

    if len(chain_endpoints) >= 2:
        return chain_endpoints[0], chain_endpoints[1]
    elif len(chain_endpoints) == 1:
        return chain_endpoints[0], None
    else:
        first_contour = contours_info[component[0]]
        last_contour = contours_info[component[-1]]
        return first_contour["p1"], last_contour["p2"]

def determine_start_end_points(component, contours_info, endpoint1, endpoint2, has_arrow_head):
    """
    Given merged component, decide which endpoint is start and which is end based on
    original arrow-head location. If no arrow head present, return (None, None).
    """
    start_point = None
    end_point = None

    if not has_arrow_head:
        return None, None

    arrow_head_contour = None
    for idx in component:
        if contours_info[idx]["end"] is not None:
            arrow_head_contour = contours_info[idx]
            break

    if arrow_head_contour and arrow_head_contour["end"] is not None:
        arrow_head_location = arrow_head_contour["end"]

        dist_to_endpoint1 = math.dist(endpoint1, arrow_head_location) if endpoint1 else float('inf')
        dist_to_endpoint2 = math.dist(endpoint2, arrow_head_location) if endpoint2 else float('inf')

        if dist_to_endpoint1 < dist_to_endpoint2:
            end_point = endpoint1
            start_point = endpoint2
        else:
            end_point = endpoint2
            start_point = endpoint1
    else:
        start_point = endpoint1
        end_point = endpoint2

    return start_point, end_point

# ---------------- Step 6: OCR and Text Processing ----------------
def detect_separator_lines_in_rectangle(rectangle, image):
    """
    Detect vertical separator lines inside a class rectangle to split name/attributes/methods.
    Returns two Y coordinates (line1_y, line2_y). If detection fails, heuristics are used.
    """
    x1, y1, x2, y2 = rectangle
    rect_roi = image[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(rect_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vertical_lines = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    line_contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    separator_lines = []
    for cnt in line_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > (y2 - y1) * 0.3 and w < 10:  # tall and thin
            global_y = y1 + y
            separator_lines.append(global_y)

    if len(separator_lines) >= 2:
        separator_lines.sort()
        line1 = separator_lines[0]
        line2 = separator_lines[1]
        return line1, line2
    else:
        rect_height = y2 - y1
        line1 = y1 + rect_height * 0.2
        line2 = y1 + rect_height * 0.6
        return line1, line2

def separate_access_modifier(text):
    """
    Separate access modifier symbol (+, -, #, ~) from the rest of a text line.
    Defaults to public (+) when none is found.
    """
    access_modifiers = ['+', '-', '#', '~']
    for modifier in access_modifiers:
        if text.startswith(modifier):
            return modifier, text[len(modifier):].strip()
    return '+', text

def process_text_lines_with_modifiers(text_items):
    """
    Group OCR text items into logical lines, sort by X, and separate access modifiers.
    Returns a list of [modifier, content] per line.
    """
    if not text_items:
        return []

    # Sort by top Y then X (approx reading order)
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
            if current_line:
                current_line.sort(key=lambda item: item['x'])
                line_text = ' '.join(item['text'] for item in current_line)
                modifier, content = separate_access_modifier(line_text)
                lines.append([modifier, content])
            current_line = [item]
            current_line_y = y_pos

    if current_line:
        current_line.sort(key=lambda item: item['x'])
        line_text = ' '.join(item['text'] for item in current_line)
        modifier, content = separate_access_modifier(line_text)
        lines.append([modifier, content])

    return lines

def find_closest_class(classes, x, y):
    """
    Return the class name whose rectangle is closest to the given point (x, y).
    """
    closest_class = None
    min_distance = float('inf')

    for class_info in classes:
        rect_x1, rect_y1, rect_x2, rect_y2 = class_info['position']
        dx = max(rect_x1 - x, 0, x - rect_x2)
        dy = max(rect_y1 - y, 0, y - rect_y2)
        distance = math.sqrt(dx*dx + dy*dy)
        if distance < min_distance:
            min_distance = distance
            closest_class = class_info['name']

    return closest_class

def normalize_relationship_type(relationship_type):
    """
    Normalize relationship labels to canonical names.
    """
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
    """
    Identify multiplicity-like OCR texts (e.g., 0..1, 1..*, *) and associate them
    to the closest relationship and class.
    """
    multiplicities = []
    multiplicity_patterns = ['0..1', '1..*', '0..*', '1', '*', '0..n', '1..n', '..']

    for text_item in outside_texts:
        text = text_item['text']
        x, y = text_item['x'], text_item['y']

        is_multiplicity = any(pattern in text for pattern in multiplicity_patterns)

        if is_multiplicity:
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
    """
    Return the index of the relationship contour that is closest to point (x, y),
    using the line segment midpoint as representative location.
    """
    closest_relationship = None
    min_distance = float('inf')

    for i, cont in enumerate(contours):
        if cont['start'] and cont['end']:
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
    """
    Return the index of the rectangle (class) whose center is closest to point (x, y).
    """
    closest_class = None
    min_distance = float('inf')

    for i, rect in enumerate(rectangles):
        rx1, ry1, rx2, ry2 = rect
        rect_center_x = (rx1 + rx2) / 2
        rect_center_y = (ry1 + ry2) / 2
        distance = math.sqrt((x - rect_center_x)**2 + (y - rect_center_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_class = i

    return closest_class

def find_multiplicity_for_endpoint(multiplicities, relationship_index, contour, endpoint):
    """
    Given multiplicities and a relationship contour, find the multiplicity nearest to the contour endpoint.
    """
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
    """
    Build relationship structures between classes and attach multiplicities where available.
    """
    relationships = []

    for i, cont in enumerate(contours):
        relationship_type = cont['classification'].lower()

        # Skip generic association lines unless they carry additional info
        if 'association' in relationship_type and relationship_type == 'association':
            continue

        start_class = None
        end_class = None

        if cont['start'] and cont['end']:
            start_x, start_y = cont['start']
            end_x, end_y = cont['end']
            start_class = find_closest_class(classes, start_x, start_y)
            end_class = find_closest_class(classes, end_x, end_y)

        # Fallback: use p1/p2 if start/end not available
        if not start_class and cont['p1']:
            start_x, start_y = cont['p1']
            start_class = find_closest_class(classes, start_x, start_y)
        if not end_class and cont['p2']:
            end_x, end_y = cont['p2']
            end_class = find_closest_class(classes, end_x, end_y)

        if start_class and end_class and start_class != end_class:
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
    """
    Convert OCR output and detected shapes into a structured representation:
    - classes: list of {name, attributes, methods, position}
    - relationships: list of relationships with multiplicities attached
    """
    structuredData = {
        'classes': [],
        'relationships': []
    }

    # Separate OCR results into texts inside rectangles (class content) and outside
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

        text_inside_any_rect = False
        for rect in rectangles:
            rx1, ry1, rx2, ry2 = rect
            if (x >= rx1 and y >= ry1 and (x + w) <= rx2 and (y + h) <= ry2):
                text_inside_any_rect = True
                break

        if text_inside_any_rect:
            inside_texts.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h})
        else:
            outside_texts.append({'text': text, 'x': x, 'y': y, 'w': w, 'h': h})

    # Process each rectangle as a class (use separator lines to split name/attributes/methods)
    for i, rect in enumerate(rectangles):
        x1, y1, x2, y2 = rect

        line1_y, line2_y = detect_separator_lines_in_rectangle(rect, original_image)

        class_name_texts = []
        attribute_texts = []
        method_texts = []

        for text_item in inside_texts:
            x = text_item['x']
            y = text_item['y']
            w = text_item['w']
            h = text_item['h']
            text = text_item['text']

            if (x >= x1 and y >= y1 and (x + w) <= x2 and (y + h) <= y2):
                text_center_y = y + h / 2

                if text_center_y < line1_y:
                    class_name_texts.append({'y': y, 'text': text, 'x': x})
                elif line1_y <= text_center_y <= line2_y:
                    attribute_texts.append({'y': y, 'text': text, 'x': x})
                else:
                    method_texts.append({'y': y, 'text': text, 'x': x})

        class_name = f"Class{i+1}"
        if class_name_texts:
            class_name_texts.sort(key=lambda item: (item['y'], item['x']))
            class_name = ' '.join(item['text'] for item in class_name_texts)

        attributes = process_text_lines_with_modifiers(attribute_texts)
        methods = process_text_lines_with_modifiers(method_texts)

        class_info = {
            'name': class_name,
            'attributes': attributes,
            'methods': methods,
            'position': (x1, y1, x2, y2)
        }
        structuredData['classes'].append(class_info)

    # Identify multiplicities from texts outside classes
    multiplicities = process_multiplicities(outside_texts, contours, rectangles)

    # Build relationships and attach multiplicities
    structuredData['relationships'] = process_relationships_with_multiplicities(
        contours, structuredData['classes'], multiplicities
    )

    return structuredData

# ---------------- Main workflow ----------------
def main(image_path):
    """
    Full pipeline entry point. Returns structuredData for the diagram image.
    """
    image, gray = load_image(image_path)

    # Initial threshold and line extraction
    thresh = threshold_image(gray)
    vertical_lines, horizontal_lines = extract_lines(thresh)

    # Segment vertical & horizontal lines into bounding boxes
    vertical_segments = get_segments(vertical_lines)
    horizontal_segments = get_segments(horizontal_lines)

    # Pair aligned vertical/horizontal segments
    vertical_pairs = find_vertical_pairs(vertical_segments)
    horizontal_pairs = find_horizontal_pairs(horizontal_segments)

    # Detect rectangles formed by paired lines and the line areas that compose them
    rectangles, lines_to_remove = detect_rectangles(vertical_pairs, horizontal_pairs, vertical_segments, horizontal_segments)

    # Remove rectangle line areas from the original image (so relationships remain)
    image_rect = draw_rectangles(image, rectangles, lines_to_remove)

    # Re-threshold after rectangle removal
    noRectImg = image_rect.copy()
    threshImg = threshold_image(cv2.cvtColor(noRectImg, cv2.COLOR_BGR2GRAY))

    # Detect, classify, and merge contours (relationships)
    contours, hierarchy, detectedContours, drawContour, textImg = detect_contours(threshImg, rectangles)

    # OCR on textImg and convert to BGR for drawing/consistency
    data = pytesseract.image_to_data(textImg, output_type=Output.DICT)
    textImg = cv2.cvtColor(textImg, cv2.COLOR_GRAY2BGR)

    structuredData = reStructureData(rectangles, contours, data, textImg)
    print(structuredData)
    
    return structuredData
   

# ---------------- Run ----------------
import time

if __name__ == "__main__":
    # Start timing
    start_time = time.time()

    # Example run (keeps original behavior)
    main("./images/cd_comp.png")

    # End timing
    end_time = time.time()

    # Calculate and print runtime
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.4f} seconds")
