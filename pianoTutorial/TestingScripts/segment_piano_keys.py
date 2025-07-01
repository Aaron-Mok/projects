from ultralytics import YOLO
import cv2

# Load your trained model
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

    frame = cv2.flip(frame, 0)
    # Predict segmentation
    results = model(frame, conf=0.5)[0]

    # Draw segmentation mask
    annotated = results.plot()
    display_image = cv2.resize(annotated, (0, 0), fx=rescale, fy=rescale)
    cv2.imshow("Piano Segmentation", display_image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()