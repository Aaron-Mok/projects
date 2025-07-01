from ultralytics import YOLO
import cv2
import mediapipe as mp
import numpy as np
import torch

#TODO:Add rotation

# Load your trained YOLOv8 segmentation model
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("piano_seg_yolov8/runs/segment/train2/weights/best.pt").to(device)

# Set up MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False,
#                        max_num_hands=2,
#                        min_detection_confidence=0.7,
#                        min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

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
    results = model(frame, conf=0.8)[0]

    annotated = frame.copy()  # we’ll draw on this

    # === Try to extract piano mask and crop ===
    if results.masks is not None and len(results.masks.data) > 0:
        mask = results.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        x, y, w_box, h_box = cv2.boundingRect(mask_resized)

        # Expand vertically to include full white keys
        # pad_y = int(h_box * 1.5)
        # y1 = max(y - pad_y, 0)
        # y2 = min(y + h_box + pad_y, frame.shape[0])
        y1 = y
        y2 = y + h_box 
        x1 = x + 8
        x2 = x + w_box

        keyboard_crop = frame[y1:y2, x1:x2]

        # === Divide into 52 white keys ===
        NUM_KEYS = 52
        trim_left = 0   # <-- adjust this if needed
        trim_right = 0  # <-- adjust this if needed
        h_crop, w_crop, _ = keyboard_crop.shape
        left = trim_left
        right = w_crop - trim_right
        cropped_width = right - left
        key_width = cropped_width // NUM_KEYS

        for i in range(NUM_KEYS):
            kx1 = left + i * key_width
            kx2 = kx1 + key_width
            ky1 = 0
            ky2 = h_crop
            cv2.rectangle(keyboard_crop, (kx1, ky1), (kx2, ky2), (0, 255, 0), 1)
            cv2.putText(keyboard_crop, f"{i+1}", (kx1 + 5, ky2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Show divided keys (in a separate window)
        resize_frame = cv2.resize(frame, (0, 0), fx=rescale, fy=rescale)
        cv2.imshow("Divided 52 White Keys", resize_frame)

    # fallback if no mask: show frame only
    annotated = results.plot()

    # MediaPipe finger tracking
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # result = hands.process(rgb)

    # h, w, _ = frame.shape
    # if result.multi_hand_landmarks:
    #     for hand_landmarks in result.multi_hand_landmarks:
    #         mp_draw.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #         for idx in [4, 8, 12, 16, 20]:  # Thumb to Pinky fingertips
    #             lm = hand_landmarks.landmark[idx]
    #             cx, cy = int(lm.x * w), int(lm.y * h)
    #             cv2.circle(annotated, (cx, cy), 10, (0, 255, 0), -1)
    #             cv2.putText(annotated, f"{idx}", (cx + 10, cy - 10),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # # Show combined result
    # display_image = cv2.resize(annotated, (0, 0), fx=rescale, fy=rescale)
    # cv2.imshow("Piano Segmentation + Finger Tracking", display_image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

    if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save
        cv2.imwrite("saved_frame.jpg", frame)
        print("Image saved as saved_frame.jpg")


cap.release()
cv2.destroyAllWindows()
