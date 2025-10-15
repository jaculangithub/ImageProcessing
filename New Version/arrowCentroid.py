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
                        cv2.drawContours(blank_image, [cnt], -1, 255, 1)

        return blank_image
    else:
        return None


def get_length(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def angle_beween_points(a, b):
    arrow_slope = (a[0] - b[0]) / (a[1] - b[1] + 1e-5)  # Avoid division by zero
    arrow_angle = math.degrees(math.atan(arrow_slope))
    return arrow_angle


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


def get_arrow_info(arrow_image):
    arrow_info_image = cv2.cvtColor(arrow_image.copy(), cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(arrow_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrow_info = []
    
    if hierarchy is not None:
        for cnt in contours:
            # Find tip and tail using convex hull point density
            tip, tail, centroid = find_arrow_tip_and_tail(cnt)
            
            if tip is not None and tail is not None and centroid is not None:
                # Calculate angle and length
                angle = angle_beween_points(tip, tail)
                length = get_length(tip, tail)
                
                # Draw the arrow with tip and tail clearly marked
                cv2.line(arrow_info_image, tail, tip, (0, 255, 255), 2)  # Draw from tail to tip
                
                # Mark CENTROID with GREEN dot
                cv2.circle(arrow_info_image, centroid, 6, (0, 255, 0), -1)
                cv2.putText(arrow_info_image, "CENTROID", (centroid[0] + 10, centroid[1]), 
                           cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                
                # Mark TIP (pointy end) with RED
                cv2.circle(arrow_info_image, tip, 8, (0, 0, 255), -1)
                cv2.putText(arrow_info_image, "TIP", (tip[0] + 10, tip[1]), 
                           cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
                
                # Draw arrowhead at the TIP
                draw_arrowhead(arrow_info_image, tip, tail)
                
                # Mark TAIL/START (broad end) with BLUE
                cv2.circle(arrow_info_image, tail, 6, (255, 0, 0), -1)
                cv2.putText(arrow_info_image, "START", (tail[0] + 10, tail[1]), 
                           cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 0), 2)
                
                # Add angle and length information
                mid_point = ((tip[0] + tail[0])//2, (tip[1] + tail[1])//2)
                cv2.putText(arrow_info_image, f"Angle: {angle:.2f}°", 
                           (mid_point[0], mid_point[1] + 20), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                cv2.putText(arrow_info_image, f"Length: {length:.2f}px", 
                           (mid_point[0], mid_point[1] + 40), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                
                # Store arrow information
                arrow_info.append({
                    'tip': tip,
                    'tail': tail,
                    'centroid': centroid,
                    'angle': angle,
                    'length': length
                })

        return arrow_info_image, arrow_info
    else:
        return None, None


def draw_arrowhead(image, tip, tail, size=12):
    """Draw a proper arrowhead at the TIP"""
    # Calculate direction vector from tail to tip
    dx = tip[0] - tail[0]
    dy = tip[1] - tail[1]
    
    # Normalize
    length = math.sqrt(dx*dx + dy*dy)
    if length > 0:
        dx /= length
        dy /= length
    
    # Calculate perpendicular vector
    px = -dy
    py = dx
    
    # Calculate arrowhead points
    left_point = (int(tip[0] - size*dx + size*px), int(tip[1] - size*dy + size*py))
    right_point = (int(tip[0] - size*dx - size*px), int(tip[1] - size*dy - size*py))
    
    # Draw arrowhead
    cv2.line(image, tip, left_point, (0, 0, 255), 2)
    cv2.line(image, tip, right_point, (0, 0, 255), 2)


if __name__ == "__main__":
    image = cv2.imread("./images/no_classes1.png")

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Show threshold
    plt.imshow(thresh_image, cmap="gray")
    plt.title("Threshold Image")
    plt.axis("off")
    plt.show()

    arrow_image = get_filter_arrow_image(thresh_image)
    if arrow_image is not None:
        plt.imshow(arrow_image, cmap="gray")
        plt.title("Arrow Image")
        plt.axis("off")
        plt.show()

        arrow_info_image, arrow_info = get_arrow_info(arrow_image)

        # Convert BGR (OpenCV) to RGB (matplotlib expects RGB)
        arrow_info_rgb = cv2.cvtColor(arrow_info_image, cv2.COLOR_BGR2RGB)

        plt.imshow(arrow_info_rgb)
        plt.title("Arrow Info Image (Red=TIP, Blue=START, Green=CENTROID)")
        plt.axis("off")
        plt.show()
        
        # Print arrow information
        for i, info in enumerate(arrow_info):
            print(f"Arrow {i+1}:")
            print(f"  Tip (pointy end): {info['tip']}")
            print(f"  Start (broad end): {info['tail']}")
            print(f"  Centroid: {info['centroid']}")
            print(f"  Angle: {info['angle']:.2f}°")
            print(f"  Length: {info['length']:.2f}px")
            print()