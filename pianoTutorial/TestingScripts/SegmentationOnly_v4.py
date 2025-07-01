from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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

# frame = cv2.imread("saved_frame.jpg")  # Use a saved image for testing

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

        cv2.imshow("Gaussian blurred image", gray_crop)

        adaptive = cv2.adaptiveThreshold(
            gray_crop,                           # Grayscale input
            255,                            # Max value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Adaptive method
            cv2.THRESH_BINARY_INV,              # Normal binary (not inverted)
            11,                             # Block size (odd number, e.g., 11 or 15)
            2                               # Constant subtracted (tweakable)
        )

        cv2.imshow("Adaptive Threshold", adaptive)

        edges = cv2.Canny(gray_crop, 50, 150, apertureSize=3)

        cv2.imshow("Adaptive Threshold", edges)

        lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=5,        # adjust for sensitivity
        minLineLength=100,    # adjust for vertical/horizontal piano edges
        maxLineGap=10         # adjust for gaps between key edges
        )

        line_img = cropped.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2((y2 - y1), (x2 - x1)) * 180 / np.pi
                if abs(angle) > 80:  # keep mostly vertical lines (80° to 100°)
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Detected Hough Lines", line_img)

        

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

    # cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
        cv2.imwrite("saved_frame.jpg", frame)
        print("Image saved as saved_frame.jpg")

cap.release()
cv2.destroyAllWindows()
