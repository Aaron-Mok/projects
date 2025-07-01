import cv2
import mediapipe as mp

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Resize factor
resize_scale = 0.5

# Define piano keys: 7 white keys at the bottom of the image
NUM_KEYS = 7
KEY_NAMES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
KEY_COLOR = (255, 255, 255)
KEY_HIGHLIGHT = (0, 255, 0)

# Finger tips to track
FINGER_TIPS = {
    'Thumb': 4,
    'Index': 8,
    'Middle': 12,
    'Ring': 16,
    'Pinky': 20
}

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)

    h, w, _ = image.shape
    key_width = w // NUM_KEYS
    key_height = 150
    pressed_keys = []

    # Draw piano key layout
    for i, name in enumerate(KEY_NAMES):
        x1 = i * key_width
        x2 = x1 + key_width
        cv2.rectangle(image, (x1, h - key_height), (x2, h), KEY_COLOR, -1)
        cv2.rectangle(image, (x1, h - key_height), (x2, h), (0, 0, 0), 2)
        cv2.putText(image, name, (x1 + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Process fingers
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for finger_name, idx in FINGER_TIPS.items():
                lm = hand_landmarks.landmark[idx]
                fx = int(lm.x * w)
                fy = int(lm.y * h)
                cv2.circle(image, (fx, fy), 10, (0, 0, 255), -1)

                # Check if finger tip is on a key
                if h - key_height <= fy <= h:
                    key_index = fx // key_width
                    if 0 <= key_index < NUM_KEYS:
                        key_name = KEY_NAMES[key_index]
                        pressed_keys.append(f"{finger_name} → {key_name}")

                        # Highlight pressed key
                        x1 = key_index * key_width
                        x2 = x1 + key_width
                        cv2.rectangle(image, (x1, h - key_height), (x2, h), KEY_HIGHLIGHT, -1)
                        cv2.rectangle(image, (x1, h - key_height), (x2, h), (0, 0, 0), 2)
                        cv2.putText(image, key_name, (x1 + 10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Show pressed keys
    for i, text in enumerate(pressed_keys):
        cv2.putText(image, text, (30, 50 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize display
    display_image = cv2.resize(image, None, fx=resize_scale, fy=resize_scale)
    cv2.imshow("Piano Key Detection", display_image)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
