from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import pdb


# TODO: horizontal line in houdge line can be used for rotation
# TODO: do dilation on canny lines

def extend_line(x1, y1, x2, y2, img_width, img_height):
    if x2 - x1 == 0:
        # Vertical line
        x = x1
        return (x, 0), (x, img_height - 1)
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        y_start, y_end = 0, img_height-1
        x_start = int((y_start - intercept) / slope)
        x_end = int((y_end - intercept) / slope)
        return (x_start, y_start), (x_end, y_end)

def get_slope_intercept(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf'), x1  # Vertical line: slope = ∞, intercept is x
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def merge_similar_lines(lines, slope_thresh=0.01, intercept_thresh=1):
    merged = []
    for slope, intercept in lines:
        matched = False
        for group in merged:
            avg_slope, avg_intercept, count = group
            if abs(slope - avg_slope) < slope_thresh and abs(intercept - avg_intercept) < intercept_thresh:
                group[0] = (avg_slope * count + slope) / (count + 1)
                group[1] = (avg_intercept * count + intercept) / (count + 1)
                group[2] += 1
                matched = True
                break
        if not matched:
            merged.append([slope, intercept, 1])
    return [(s, i) for s, i, _ in merged]

def average_x1_x2(line):
    x1, y1, x2, y2 = line
    return (x1 + x2) / 2

def merge_by_x_proximity(lines, x_thresh=10):
    lines_sorted = sorted(lines, key=average_x1_x2)
    merged = []
    for line in lines_sorted:
        x = average_x1_x2(line)
        if not merged:
            merged.append([line, x, 1])
        else:
            last_avg_x = merged[-1][1]
            if abs(x - last_avg_x) < x_thresh:
                # average the lines
                prev_line, prev_x, count = merged[-1]
                new_line = [(a + b) // 2 for a, b in zip(prev_line, line)]
                merged[-1] = [new_line, (prev_x * count + x) / (count + 1), count + 1]
            else:
                merged.append([line, x, 1])
    return [entry[0] for entry in merged]

# Load your trained YOLOv8 segmentation model
model = YOLO("piano_seg_yolov8/runs/segment/train2/weights/best.pt")


# Start webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rescale = 0.5

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

frame = cv2.imread("saved_frame.jpg")  # Use a saved image for testing

    # frame = cv2.flip(frame, -1)  # Flip upside down if needed

# YOLOv8 segmentation
results = model(frame, conf=0.9)[0]
# Create base image for annotation
annotated = results.plot()

# === Refine mask ===
if results.masks is not None and len(results.masks.data) > 0:
    # Step 1: YOLO mask (resized and binarized)
    yolo_mask = results.masks.data[0].cpu().numpy()
    yolo_mask = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))
    yolo_mask = (yolo_mask * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(yolo_mask, 128, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(binary_mask)
    margin = 10

    cropped = frame[y-margin:y+h+margin, x-margin:x+w+margin]
    
    cv2.imshow("YOLO Piano Mask", cropped)

    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # gray_crop = cv2.GaussianBlur(gray_crop, (11, 11), 0)

    h = gray_crop.shape[0]
    roi = gray_crop[int(h * 0.7):]  # bottom quarter

    edges = cv2.Canny(roi, 50, 150, apertureSize=3)

    kernel = np.ones((3, 3), np.uint8)
    closed_edge = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Edge detection", edges)
    cv2.imshow("Closed edge detection", closed_edge)

    lines_vis = cv2.HoughLines(closed_edge, rho=1, theta=np.pi/180, threshold=40)
    accumulator = np.zeros(180, dtype=int)  # one per degree

    if lines_vis is not None:
        for rho_theta in lines_vis:
            rho, theta = rho_theta[0]
            deg = int(np.rad2deg(theta)) % 180
            accumulator[deg] += 1

    # 4. Plot it
    # plt.figure(figsize=(10, 4))
    # plt.title("Hough Transform Accumulator (angle histogram)")
    # plt.xlabel("Theta (degrees)")
    # plt.ylabel("Votes")
    # plt.plot(np.arange(180), accumulator)
    # plt.grid(True)
    # plt.show()

    lines = cv2.HoughLinesP(
    closed_edge,
    rho=1,
    theta=np.pi / 180,
    threshold=10,        # The minimum number of intersections to "*detect*" a line
    minLineLength=20,    # The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    maxLineGap=10        # The maximum gap between two points to be considered in the same line.
    )

    vertical_lines = []
    slopes_intercepts = []
    line_img = roi.copy()
    line_img = cv2.cvtColor(line_img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
            if abs(angle) > 80:  # keep mostly vertical lines (80° to 100°)
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vertical_lines.append((x1, y1, x2, y2))

    cv2.imshow("Hough Lines", line_img)

# print(vertical_lines[0][0])

# def get_x1(line):
#     return line[0]

# vertical_lines_sorted = sorted(vertical_lines, key=get_x1)  # x1

# merged_lines = []
# merge_thresh = 10  # adjust as needed

# for line in vertical_lines_sorted:
#     x1, y1, x2, y2 = line
#     if not merged_lines:
#         merged_lines.append((x1, y1, x2, y2))
#     else:
#         last_x1, last_y1, last_x2, last_y2 = merged_lines[-1]
#         if abs(x1 - last_x1) > merge_thresh:
#             merged_lines.append((x1, y1, x2, y2))
#         # else: skip it as a near-duplicate


#     line_img_2 = cv2.cvtColor(line_img_2, cv2.COLOR_GRAY2BGR)

#     for x1, y1, x2, y2 in merged_lines:
#         cv2.line(line_img_2, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Merge similar lines
    merged_lines = merge_by_x_proximity(vertical_lines)

    # Visualize
    line_img_2 = roi.copy()
    output = cv2.cvtColor(line_img_2, cv2.COLOR_GRAY2BGR)

    for line in merged_lines:
        x1, y1, x2, y2 = line
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Merged Hough line", output)

cropped_copy = cropped.copy()

for x1, y1, x2, y2 in merged_lines:
    y1 = int(y1 + h * 0.7)  # Adjust y1 to match the original cropped image
    y2 = int(y2 + h * 0.7)  # Adjust y2 to match the original cropped image
    pt1, pt2 = extend_line(x1, y1, x2, y2, cropped_copy.shape[1], cropped_copy.shape[0])
    # Optionally clip y to avoid out-of-bounds
    # pt1 = (pt1[0], np.clip(pt1[1], 0, cropped_copy.shape[0]-1))
    # pt2 = (pt2[0], np.clip(pt2[1], 0, cropped_copy.shape[0]-1))
    cv2.line(cropped_copy, pt1, pt2, (0, 255, 0), 2)
    
cv2.imshow("Piano line", cropped_copy)

        # kernel = np.ones((5, 5), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # adaptive_clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

        # cv2.imshow("Adaptive Threshold with opening", adaptive_clean)

        # kernel = np.ones((21,21), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # adaptive_clean = cv2.morphologyEx(adaptive_clean, cv2.MORPH_CLOSE, kernel)

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # adaptive_clean = cv2.morphologyEx(adaptive_clean, cv2.MORPH_OPEN, kernel)

        # cv2.imshow("Adaptive Threshold with opening then closing", adaptive_clean)

        # ys, xs = np.where(adaptive_clean > 0)
        # points = np.column_stack((xs, ys))  # Shape: (N, 2)

        # hull = cv2.convexHull(points)

        # # 2. Approximate hull with 4 points
        # epsilon = 0.05 * cv2.arcLength(hull, True)
        # approx = cv2.approxPolyDP(hull, epsilon, True)

        # # 3. If not 4 points, increase epsilon until it becomes 4
        # while len(approx) > 4 and epsilon < 0.1 * cv2.arcLength(hull, True):
        #     epsilon += 1
        #     approx = cv2.approxPolyDP(hull, epsilon, True)

        # # 4. If 4-point result found, draw it
        # if len(approx) == 4:
        #     trapezium = approx.reshape(4, 2)
        #     vis = cv2.cvtColor(adaptive_clean, cv2.COLOR_GRAY2BGR)
        #     cv2.polylines(vis, [trapezium], isClosed=True, color=(0, 255, 0), thickness=2)
        #     cv2.imshow("Trapezium Fit", vis)
        # else:
        #     print("Failed to find 4-point trapezium.")


    # # Show combined result
    # display_image = cv2.resize(annotated, (0, 0), fx=rescale, fy=rescale)
    # cv2.imshow("Piano Segmentation + Finger Tracking", display_image)

cv2.waitKey(0)

    # if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
    #     break

    # if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
    #     cv2.imwrite("saved_frame.jpg", frame)
    #     print("Image saved as saved_frame.jpg")

# cap.release()
cv2.destroyAllWindows()