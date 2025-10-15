import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_filter_arrow_image(threslold_image, min_area=200):
    blank_image = np.zeros_like(threslold_image)

    # dilate image to remove self-intersections error
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    threslold_image = cv2.dilate(threslold_image, kernel_dilate, iterations=1)

    contours, hierarchy = cv2.findContours(threslold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is not None:
        threshold_distnace = 1000

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:   # 🚫 skip tiny contours
                continue

            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)

            if defects is not None:
                for i in range(defects.shape[0]):
                    start_index, end_index, farthest_index, distance = defects[i, 0]

                    if distance > threshold_distnace:
                        cv2.drawContours(blank_image, [cnt], -1, 255, -1)

        return blank_image
    else:
        return None


def get_length(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def angle_beween_points(a, b):
    arrow_slope = (a[0] - b[0]) / (a[1] - b[1] + 1e-5)  # Avoid division by zero
    arrow_angle = math.degrees(math.atan(arrow_slope))
    return arrow_angle


def determine_direction(tip, tail):
    """
    Determine where the arrow points based on tip and tail positions
    Returns: 'left', 'right', 'top', 'bottom'
    """
    dx = tip[0] - tail[0]  # x difference (tip - tail)
    dy = tip[1] - tail[1]  # y difference (tip - tail)
    
    # Calculate the absolute angle from horizontal
    angle = math.degrees(math.atan2(dy, dx))
    
    # Normalize angle to 0-360 range
    if angle < 0:
        angle += 360
    
    # Determine direction based on angle ranges
    if (angle >= 315 and angle <= 360) or (angle >= 0 and angle < 45):
        return 'right'
    elif angle >= 45 and angle < 135:
        return 'bottom'  # Note: In image coordinates, positive Y is downward
    elif angle >= 135 and angle < 225:
        return 'left'
    else:  # angle >= 225 and angle < 315
        return 'top'


def find_arrow_tip_and_tail(cnt):
    """
    Find the REAL tip and tail using convex hull point density.
    TIP = point with more nearby convex hull points (complex arrowhead shape)
    TAIL = point with fewer nearby convex hull points (simpler shape)
    """
    # Get convex hull points
    hull_points = cv2.convexHull(cnt)
    
    # Find the two farthest points in the contour
    max_distance = 0
    point1, point2 = None, None
    
    for [[x1, y1]] in cnt:
        for [[x2, y2]] in cnt:
            distance = get_length((x1, y1), (x2, y2))
            if distance > max_distance:
                max_distance = distance
                point1 = (x1, y1)
                point2 = (x2, y2)
    
    if point1 is None or point2 is None:
        return None, None, None
    
    # Calculate centroid for reference
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None, None, None
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = (cx, cy)
    
    # Count how many convex hull points are near each candidate point
    threshold_distance = 30  # Adjust this based on arrow size
    
    count_near_point1 = 0
    count_near_point2 = 0
    
    for point in hull_points:
        hull_point = tuple(point[0])
        dist1 = get_length(point1, hull_point)
        dist2 = get_length(point2, hull_point)
        
        if dist1 < threshold_distance:
            count_near_point1 += 1
        if dist2 < threshold_distance:
            count_near_point2 += 1
    
    # The point with MORE nearby convex hull points is the TIP (complex arrowhead)
    # The point with FEWER nearby convex hull points is the TAIL (simpler shape)
    if count_near_point1 > count_near_point2:
        tip = point1
        tail = point2
    else:
        tip = point2
        tail = point1
    
    return tip, tail, centroid


def detect_arrows(arrow_image):
    """
    Detect arrows and return an array of arrow objects with direction information
    """
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrows = []
    
    if hierarchy is not None:
        for cnt in contours:
            # Find tip and tail using convex hull point density
            tip, tail, centroid = find_arrow_tip_and_tail(cnt)
            
            if tip is not None and tail is not None and centroid is not None:
                # Calculate angle and length
                angle = angle_beween_points(tip, tail)
                length = get_length(tip, tail)
                
                # Determine direction
                direction = determine_direction(tip, tail)
                
                # Create arrow object
                arrow_obj = {
                    'tip': tip,           # Pointy end (x, y)
                    'tail': tail,         # Starting point (x, y)
                    'centroid': centroid, # Center of mass (x, y)
                    'angle': angle,       # Angle in degrees
                    'length': length,     # Length in pixels
                    'direction': direction # Where it points: 'left', 'right', 'top', 'bottom'
                }
                
                arrows.append(arrow_obj)
    
    return arrows


if __name__ == "__main__":
    image = cv2.imread("./images/no_classes1.png")
    if image is None:
        raise FileNotFoundError("Image not found: ./images/no_rect.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    
    arrow_image = get_filter_arrow_image(thresh_image)
    
    if arrow_image is not None:
        # Detect arrows and get array of arrow objects
        arrows = detect_arrows(arrow_image)
        
        # --- VISUALIZATION: draw contour and mark TIP / TAIL / CENTROID on a copy of original image ---
        vis = image.copy()
        # draw detected contours (from arrow mask)
        contours_vis, _ = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_vis, -1, (0, 255, 0), 2)  # green contours

        # draw each arrow markers
        for idx, arrow in enumerate(arrows):
            tip = arrow['tip']
            tail = arrow['tail']
            centroid = arrow['centroid']

            # Draw line from tail to tip (yellow)
            cv2.line(vis, tail, tip, (0, 255, 255), 2)

            # Tail (blue)
            cv2.circle(vis, tail, 6, (255, 0, 0), -1)
            cv2.putText(vis, "TAIL", (tail[0] + 8, tail[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 2)

            # Tip (red)
            cv2.circle(vis, tip, 6, (0, 0, 255), -1)
            cv2.putText(vis, "TIP", (tip[0] + 8, tip[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)

            # Centroid (green)
            cv2.circle(vis, centroid, 5, (0, 255, 0), -1)
            cv2.putText(vis, "C", (centroid[0] + 8, centroid[1] + 4),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), 2)

        # Convert BGR → RGB for Matplotlib and show
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_rgb)
        plt.title("Detected arrows — contour (green), TIP (red), TAIL (blue), Centroid (green)")
        plt.axis("off")
        plt.show()

        # Print arrow information
        print(f"Detected {len(arrows)} arrow(s):")
        print("=" * 50)
        for i, arrow in enumerate(arrows):
            print(f"Arrow {i + 1}:")
            print(f"  Start (tail): {arrow['tail']}")
            print(f"  End (tip): {arrow['tip']}")
            print(f"  Centroid: {arrow['centroid']}")
            print(f"  Angle: {arrow['angle']:.2f}°")
            print(f"  Length: {arrow['length']:.2f} pixels")
            print(f"  Direction: {arrow['direction'].upper()}")
            print()
    else:
        print("No arrows detected")
