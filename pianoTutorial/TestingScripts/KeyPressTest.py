import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Finger tip indices in MediaPipe
fingertip_indices = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]

# State buffers for velocity and debounce
prev_ys = {}  # {(hand_idx, finger_idx): prev_y}
last_press_times = {}  # {(hand_idx, finger_idx): timestamp}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, -1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    key_zone_y = int(h * 0.5)
    cv2.line(frame, (0, key_zone_y), (w, key_zone_y), (0, 255, 0), 2)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for finger_idx in fingertip_indices:
                landmark = hand_landmarks.landmark[finger_idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Visual marker for each fingertip
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

                key = (hand_idx, finger_idx)
                dy = 0
                if key in prev_ys:
                    dy = y - prev_ys[key]
                prev_ys[key] = y

                # Press detection: downward motion + entry into key zone
                if dy > 7 and y > key_zone_y:
                    last_time = last_press_times.get(key, 0)
                    if time.time() - last_time > 0.5:  # debounce
                        print(f"🎹 Key pressed by hand {hand_idx}, finger {finger_idx}")
                        cv2.putText(frame, "Key Pressed!", (x - 30, y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        last_press_times[key] = time.time()

    cv2.imshow("All Fingers Key Press Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()