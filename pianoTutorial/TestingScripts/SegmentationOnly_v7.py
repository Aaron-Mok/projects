## Created by Aaron Mok
## v7 First working version of piano segmentation with YOLO and piano key detection

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

# Load your trained YOLOv8 segmentation model
model = YOLO("piano_seg_yolov8/runs/segment/train2/weights/best.pt")

# define parameters
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rescale = 0.5
crop_margin = 10
merge_thresh = 10
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


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)  # Flip upside down if needed

    # YOLOv8 segmentation
    results = model(frame, conf=0.9)[0]
    annotated = results.plot()

    cv2.imshow("Original frame and YOLO mask", annotated)

    if results.masks is not None and len(results.masks.data) > 0:
        # Step 1: YOLO mask (resized and binarized)
        yolo_mask = results.masks.data[0].cpu().numpy()
        yolo_mask = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))
        yolo_mask = (yolo_mask * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(yolo_mask, 128, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(binary_mask)
        piano = frame[y-crop_margin:y+h+crop_margin, x-crop_margin:x+w+crop_margin]
        cv2.imshow("Piano", piano)

        piano_gray = cv2.cvtColor(piano, cv2.COLOR_BGR2GRAY)
        h = piano_gray.shape[0]
        white_keys = piano_gray[int(h * 0.7):]  # bottom quarter

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # This divide the images into 8x8 tiles and equalizes the histogram (enhance contrast by redistributing the image intensity) of each tile separately
        equalized_white_keys = clahe.apply(white_keys)
        cv2.imshow("Equalized white keys", equalized_white_keys)

        white_key_edges = cv2.Canny(equalized_white_keys, 50, 150, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        closed_edge = cv2.morphologyEx(white_key_edges, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Edge detection", white_key_edges)
        cv2.imshow("Closed edge detection", closed_edge)

        # lines_vis = cv2.HoughLines(closed_edge, rho=1, theta=np.pi/180, threshold=40)
        # accumulator = np.zeros(180, dtype=int)  # one per degree

        # if lines_vis is not None:
        #     for rho_theta in lines_vis:
        #         rho, theta = rho_theta[0]
        #         deg = int(np.rad2deg(theta)) % 180
        #         accumulator[deg] += 1

        lines = cv2.HoughLinesP(closed_edge, rho=1,theta=np.pi / 180, threshold=10, minLineLength=20, maxLineGap=10)
        # The minimum number of intersections to "*detect*" a line
        # The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
        # The maximum gap between two points to be considered in the same line.

        vertical_lines = []
        line_img = white_keys.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
                if abs(angle) > 80:  # keep mostly vertical lines (80° to 100°)
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    vertical_lines.append((x1, y1, x2, y2))
        cv2.imshow("Hough Lines", line_img)

        x_centers = [int((x1 + x2) / 2) for (x1, y1, x2, y2) in vertical_lines]
        x_centers.sort()

        merged_xs = []
        for x in x_centers:
            if not merged_xs or abs(x - merged_xs[-1]) > merge_thresh:
                merged_xs.append(x)

        # Visualize
        merged_line_img = piano.copy()
        for x in merged_xs:
            cv2.line(merged_line_img, (x, 0), (x, merged_line_img.shape[0]), (0, 255, 0), 2)

        cv2.imshow("Pianolines", merged_line_img)

    cropped_copy = piano.copy()
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

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
        cv2.imwrite("saved_frame.jpg", frame)
        print("Image saved as saved_frame.jpg")

    # cap.release()
cv2.destroyAllWindows()