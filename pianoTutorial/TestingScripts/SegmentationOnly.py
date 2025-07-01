from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
    mask = results.masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = (mask * 255).astype(np.uint8)

    # Resize mask to match original image size
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Apply mask to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_resized)

    # Collapse along rows (vertical sum) to get 1D profile
    collapsed = np.sum(masked_gray, axis=0).astype(np.float32)
    collapsed = gaussian_filter1d(collapsed, sigma=10)

    # Normalize to [0, 1]
    collapsed /= collapsed.max()
    gradient = np.diff(collapsed)

    plt.figure(figsize=(12, 4))
    plt.plot(collapsed, color='blue')
    plt.title("Normalized Horizontal Intensity Profile (YOLO Masked)")
    plt.xlabel("Horizontal Pixel Position")
    plt.ylabel("Normalized Intensity")
    plt.grid(True)
    plt.show()



# Show combined result
display_image = cv2.resize(annotated, (0, 0), fx=rescale, fy=rescale)
cv2.imshow("Piano Segmentation + Finger Tracking", display_image)

cv2.waitKey(0)

# if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
#     break

# if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
#     cv2.imwrite("saved_frame.jpg", frame)
#     print("Image saved as saved_frame.jpg")

# cap.release()
cv2.destroyAllWindows()
