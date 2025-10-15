# Install first (if not installed):
# pip install google-cloud-vision opencv-python

import io
import cv2
import numpy as np
from google.cloud import vision

# --- GOOGLE VISION: TEXT + OBJECT DETECTION ---
def detect_text_and_objects(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Request OCR + Object detection
    response = client.annotate_image({
        'image': image,
        'features': [
            {'type': vision.Feature.Type.TEXT_DETECTION},
            {'type': vision.Feature.Type.OBJECT_LOCALIZATION}
        ]
    })

    results = {}

    # Extract text
    if response.text_annotations:
        results["text"] = [t.description for t in response.text_annotations]
    else:
        results["text"] = []

    # Extract objects
    if response.localized_object_annotations:
        results["objects"] = [
            {"name": obj.name, "score": obj.score}
            for obj in response.localized_object_annotations
        ]
    else:
        results["objects"] = []

    return results


# --- OPENCV: SHAPES + LINES ---
def detect_shapes_and_lines(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # --- Detect Lines ---
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=10)
    line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)
            line_count += 1

    # --- Detect Contours (Shapes) ---
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_count = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        elif len(approx) > 6:
            shape = "Circle"
        else:
            continue
        cv2.drawContours(img, [approx], 0, (255,0,0), 2)
        shape_count += 1

    return {"lines_detected": line_count, "shapes_detected": shape_count, "annotated_image": img}


# --- MAIN ---
if __name__ == "__main__":
    image_path = "./images/atmwithdrawalv2.png"  # <<-- change this

    # Google Vision API
    vision_results = detect_text_and_objects(image_path)
    print("Text Detected:", vision_results["text"])
    print("Objects Detected:", vision_results["objects"])

    # OpenCV
    cv_results = detect_shapes_and_lines(image_path)
    print("Shapes:", cv_results["shapes_detected"])
    print("Lines:", cv_results["lines_detected"])

    # Show annotated image with shapes/lines
    cv2.imshow("Detected Shapes & Lines", cv_results["annotated_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
