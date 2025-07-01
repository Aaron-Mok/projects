from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import pdb
from PIL import Image, ImageDraw, ImageFont

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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rescale = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

# frame = cv2.imread("saved_frame_night.jpg")  # Use a saved image for testing

    frame = cv2.flip(frame, -1)  # Flip upside down if needed

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

        adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        cv2.imshow("Adaptive Threshold", adaptive)

        kernel = np.ones((3, 3), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        adaptive_clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)

        cv2.imshow("Adaptive Threshold with opening", adaptive_clean)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        equalized = clahe.apply(roi)

        cv2.imshow("CLAHE Equalized", equalized)

        edges = cv2.Canny(equalized, 50, 150, apertureSize=3)

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

        x_centers = [int((x1 + x2) / 2) for (x1, y1, x2, y2) in vertical_lines]
        x_centers.sort()

        # Merge close x values
        merged_xs = []
        merge_thresh = 10
        for x in x_centers:
            if not merged_xs or abs(x - merged_xs[-1]) > merge_thresh:
                merged_xs.append(x)


        # Merge similar lines
        # merged_lines = merge_by_x_proximity(vertical_lines)

        # Visualize
        line_img_2 = roi.copy()
        output = cv2.cvtColor(line_img_2, cv2.COLOR_GRAY2BGR)

        for x in merged_xs:
            cv2.line(output, (x, 0), (x, line_img_2.shape[0]), (0, 255, 0), 2)

    cv2.imshow("Merged Hough line", output)

    cropped_copy = cropped.copy()
    # for x1, y1, x2, y2 in merged_lines:
    #     y1 = int(y1 + h * 0.7)  # Adjust y1 to match the original cropped image
    #     y2 = int(y2 + h * 0.7)  # Adjust y2 to match the original cropped image
    #     pt1, pt2 = extend_line(x1, y1, x2, y2, cropped_copy.shape[1], cropped_copy.shape[0])
    #     # Optionally clip y to avoid out-of-bounds
    #     # pt1 = (pt1[0], np.clip(pt1[1], 0, cropped_copy.shape[0]-1))
    #     # pt2 = (pt2[0], np.clip(pt2[1], 0, cropped_copy.shape[0]-1))
    #     cv2.line(cropped_copy, pt1, pt2, (0, 255, 0), 2)

    for x in merged_xs:
        cv2.line(cropped_copy, (x, 0), (x, cropped_copy.shape[0]), (0, 255, 0), 2)

    cv2.imshow("Pianolines", cropped_copy)
    cv2.imwrite("Pianolines.png", cropped_copy)


    notes = [
        'A0', 'B0',
        'C1', 'D1', 'E1', 'F1', 'G1', 'A1', 'B1',
        'C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2',
        'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
        'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
        'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
        'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6',
        'C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7',
        'C8'
    ]


    black_notes_ascii = [
        "Bb0", "C#1", "Eb1", "F#1", "Ab1",
        "Bb1", "C#2", "Eb2", "F#2", "Ab2",
        "Bb2", "C#3", "Eb3", "F#3", "Ab3",
        "Bb3", "C#4", "Eb4", "F#4", "Ab4",
        "Bb4", "C#5", "Eb5", "F#5", "Ab5",
        "Bb5", "C#6", "Eb6", "F#6", "Ab6",
        "Bb6", "C#7", "Eb7", "F#7", "Ab7",
        "Bb7", "C#8"
    ]

    black_key_indices = [
        1, 3, 4, 6, 7, 8, 10, 11, 13, 14,
        15, 17, 18, 20, 21, 22, 24, 25, 27, 28,
        29, 31, 32, 34, 35, 36, 38, 39, 41, 42,
        43, 45, 46, 48, 49, 50
    ]

    y_pos_white = int(cropped_copy.shape[0] * 0.9)
    y_pos_black = int(cropped_copy.shape[0] * 0.6)

    if len(merged_xs) - 1 == len(notes):
        for i in range(len(merged_xs) - 1):
            mid_x = int((merged_xs[i] + merged_xs[i + 1]) / 2)
            text_size, baseline = cv2.getTextSize(notes[i], cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
            xpos = mid_x - text_size[0] // 2
            cv2.putText(cropped_copy, notes[i], (xpos, y_pos_white), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
    else:
        print("Mismatch in number of keys, adjusting note list...")


    for idx, note in zip(black_key_indices, black_notes_ascii):
        if idx < len(merged_xs) - 1:
            mid_x = int(merged_xs[idx])
            text_size, _ = cv2.getTextSize(note, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
            xpos = mid_x - text_size[0] // 2
            cv2.putText(cropped_copy, note, (xpos, y_pos_black), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 165, 255), 1, cv2.LINE_AA)

    cv2.imshow("Piano Notes", cropped_copy)

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

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
        cv2.imwrite("saved_frame.jpg", frame)
        print("Image saved as saved_frame.jpg")

    # cap.release()
cv2.destroyAllWindows()