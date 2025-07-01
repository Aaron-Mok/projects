from ultralytics import YOLO
import cv2
import mediapipe as mp

# Load your trained YOLOv8 segmentation model
model = YOLO("piano_seg_yolov8/runs/segment/train2/weights/best.pt")

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rescale = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)  # Flip upside down if needed

    # YOLOv8 segmentation
    results = model(frame, conf=0.5)[0]
    annotated = results.plot()

    # MediaPipe finger tracking
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for idx in [4, 8, 12, 16, 20]:  # Thumb to Pinky fingertips
                lm = hand_landmarks.landmark[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), -1)
                cv2.putText(annotated, f"{idx}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show combined result
    display_image = cv2.resize(annotated, (0, 0), fx=rescale, fy=rescale)
    cv2.imshow("Piano Segmentation + Finger Tracking", display_image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
        cv2.imwrite("saved_frame.jpg", frame)
        print("Image saved as saved_frame.jpg")

cap.release()
cv2.destroyAllWindows()
