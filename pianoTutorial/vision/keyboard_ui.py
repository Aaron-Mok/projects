# vision/keyboard_ui.py

import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import mediapipe as mp
import logging
from vision.constants import white_notes, black_notes, black_key_indices, note_colors

class KeyboardUI:
    def __init__(self):
        print("✅ Keyboard UI initialized")
        self.model = YOLO("piano_seg_yolov8/runs/segment/train2/weights/best.pt")
        LOGGER.setLevel(logging.CRITICAL)

        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.locked = False
        self.locked_merged_xs = None
        self.locked_crop_coords = None
        self.current_note = None
        self.last_flash_time = 0
        self.crop_margin = 10
        self.buffer_size = 10

        self.notes = white_notes
        self.black_notes = black_notes
        self.black_indices = black_key_indices 
        self.note_colors = note_colors

        self.font = ImageFont.truetype("arialbd.ttf", size=10)  # "bd" = bold
        self.last_clean_time = time.time()
        self.clean_reference = None
        self.line_buffer = []

        self.circle_radius = 10
        self.merge_thresh = 10

        self._measure_image = None
        self._show_measure = False

    # def flash_note(self, note_name):
    #     self.current_note = note_name
    #     self.last_flash_time = time.time()

    def flash_note(self, note_dict):
        # if isinstance(note_names, str):  # If it's a single note, wrap in list
        #     note_names = [note_names]
        # note_dict = {"treble": [...], "bass": [...]}
        print(f"⚡ Flashing note(s): {note_dict}")  # Add this line
        self.current_note = note_dict
        self.last_flash_time = time.time()

    def match_brightness(self, src, ref):
        # src and ref should be grayscale images

        src_mean = np.mean(src)
        ref_mean = np.mean(ref)

        # Avoid division by zero
        if ref_mean == 0:
            return ref.copy()

        scale = src_mean / ref_mean
        adjusted = np.clip(ref.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return adjusted

    def draw_note_circle(self, draw, center_x, center_y, note, is_target=False):
        letter = ''.join(filter(str.isalpha, note))
        base_color = self.note_colors.get(letter, (128, 128, 128))
        if is_target and isinstance(self.current_note, dict):
            if note in self.current_note.get("treble", []):
                color = base_color  # Light sky blue (calm but clear)
                outline_color = (255, 255, 0)
                outline_radius_margin = 5
            elif note in self.current_note.get("bass", []):
                color = base_color  # Soft mint green (friendly, readable)
                outline_color = (255, 165, 60)
                outline_radius_margin = 5
            else:
                color = (255, 255, 255)  # Fallback
        else:
            color = base_color
            outline_color = (0,0,0)  # Black outline for non-target notes
            outline_radius_margin = 2

        radius = 10 if is_target else self.circle_radius
        outline_radius = radius + outline_radius_margin
        draw.ellipse(
            [(center_x - outline_radius, center_y - outline_radius),
            (center_x + outline_radius, center_y + outline_radius)],
            fill=outline_color
        )

        # Inner colored circle
        draw.ellipse(
            [(center_x - radius, center_y - radius),
            (center_x + radius, center_y + radius)],
            fill=color
        )

        try:
            bbox = draw.textbbox((0, 0), note, font=self.font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            w, h = self.font.getsize(note)

        draw.text((center_x - w/2, center_y - h/2), note, fill=(0, 0, 0), font=self.font)


    def toggle_lock(self, current_xs, current_crop_coords):
        self.locked = not self.locked
        if self.locked:
            self.locked_merged_xs = current_xs.copy()
            self.locked_crop_coords = current_crop_coords
            print("🔒 Locked keyboard layout.")
        else:
            self.locked_merged_xs = None
            self.locked_crop_coords = None
            print("🔓 Unlocked keyboard layout.")

    def show_measure_image(self, image):
        self._measure_image = image
        self._show_measure = True

    def show(self):
        print("📷 Showing UI...")
        while True:
            if self._show_measure and self._measure_image is not None:
                cv2.imshow("Sheet Music", self._measure_image)
                cv2.moveWindow("Sheet Music", 100, 100)

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, -1)  # Flip upside down
            frame_bgr = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_hands = self.hands.process(frame_rgb)

            # YOLOv8 segmentation
            self.model.names[0] = "piano"
            results = self.model(frame, conf=0.9)[0]
            annotated = results.plot(labels = False)
            cv2.imshow("Original frame and YOLO mask", annotated)

            h_img, w_img, _ = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if results_hands.multi_hand_landmarks is None:
                if self.clean_reference is None or time.time() - self.last_clean_time > 2:
                    self.clean_reference = frame.copy()
                    self.last_clean_time = time.time()
                    print("Updated clean reference")
            else:
                if self.clean_reference is not None:
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        xs = [int(landmark.x * w_img) for landmark in hand_landmarks.landmark]
                        ys = [int(landmark.y * h_img) for landmark in hand_landmarks.landmark]
                        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w_img)
                        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h_img)

                        # Replace the hand area with pixels from the clean reference
                        clean_reference_matched = self.match_brightness(frame, self.clean_reference)
                        frame[y_min:y_max, x_min:x_max] = clean_reference_matched[y_min:y_max, x_min:x_max]

                    # Optional: Show masked debug output
                    debug_frame = frame.copy()
                    for hand_landmarks in results_hands.multi_hand_landmarks:
                        xs = [int(landmark.x * w_img) for landmark in hand_landmarks.landmark]
                        ys = [int(landmark.y * h_img) for landmark in hand_landmarks.landmark]
                        x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w_img)
                        y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h_img)
                        cv2.rectangle(debug_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.imshow("Hand Region (debug)", debug_frame)

            if results.masks is not None and len(results.masks.data) > 0:
                # Step 1: YOLO mask (resized and binarized)
                yolo_mask = results.masks.data[0].cpu().numpy()
                yolo_mask = cv2.resize(yolo_mask, (frame.shape[1], frame.shape[0]))
                yolo_mask = (yolo_mask * 255).astype(np.uint8)
                _, binary_mask = cv2.threshold(yolo_mask, 128, 255, cv2.THRESH_BINARY)
                x, y, w, h = cv2.boundingRect(binary_mask)
                if self.locked_crop_coords:
                    x, y, w, h, x2, y2, w2, h2 = self.locked_crop_coords
                piano_YOLO = frame[y-self.crop_margin:y+h+self.crop_margin, x-self.crop_margin:x+w+self.crop_margin]
                piano_YOLO_bgr = frame_bgr[y-self.crop_margin:y+h+self.crop_margin, x-self.crop_margin:x+w+self.crop_margin]

                # piano_YOLO_gray = cv2.cvtColor(piano_YOLO, cv2.COLOR_BGR2GRAY)
                piano_YOLO_gray = piano_YOLO
                _, bright_mask = cv2.threshold(piano_YOLO_gray, 180, 255, cv2.THRESH_BINARY)
                kernel = np.ones((5, 5), np.uint8)
                closed_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x2, y2, w2, h2 = cv2.boundingRect(largest)
                    if self.locked_crop_coords:
                        x, y, w, h, x2, y2, w2, h2 = self.locked_crop_coords
                    piano = piano_YOLO_gray[y2:y2 + h2, x2:x2 + w2]
                    piano_bgr = piano_YOLO_bgr[y2:y2 + h2, x2:x2 + w2]
                    cv2.imshow("Refined Keyboard Crop", piano)

                cv2.imshow("Piano", piano)

                # piano_gray = cv2.cvtColor(piano, cv2.COLOR_BGR2GRAY)
                h = piano.shape[0]
                white_keys = piano[int(h * 0.7):]  # bottom quarter

                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  # This divide the images into 8x8 tiles and equalizes the histogram (enhance contrast by redistributing the image intensity) of each tile separately
                equalized_white_keys = clahe.apply(white_keys)
                cv2.imshow("Equalized white keys", equalized_white_keys)

                white_key_edges = cv2.Canny(equalized_white_keys, 50, 150, apertureSize=3)
                kernel = np.ones((3, 3), np.uint8)
                closed_edge = cv2.morphologyEx(white_key_edges, cv2.MORPH_CLOSE, kernel)
                cv2.imshow("Edge detection", white_key_edges)
                cv2.imshow("Closed edge detection", closed_edge)

                lines = cv2.HoughLinesP(closed_edge, rho=1,theta=np.pi / 180, threshold=10, minLineLength=20, maxLineGap=10)

                vertical_lines = []
                line_img = white_keys.copy()
                if lines is not None:
                    for line in lines:
                        x_1, y_1, x_2, y_2 = line[0]
                        angle = np.arctan2((y_2 - y_1), (x_2 - x_1)) * 180 / np.pi
                        if abs(angle) > 80:  # keep mostly vertical lines (80° to 100°)
                            cv2.line(line_img, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
                            vertical_lines.append((x_1, y_1, x_2, y_2))
                cv2.imshow("Hough Lines", line_img)

                frame_x_centers = [int((x_1 + x_2) / 2) for (x_1, y_1, x_2, y_2) in vertical_lines]
                frame_x_centers.sort()
                
                merged_xs = []
                # Append to buffer
                self.line_buffer.append(frame_x_centers)
                if len(self.line_buffer) > self.buffer_size:
                    self.line_buffer.pop(0)

                if len(self.line_buffer) == self.buffer_size:
                    all_xs = [x for frame_xs in self.line_buffer for x in frame_xs]
                    all_xs.sort()

                    # Merge close x values
                    cluster = []
                    for xs in all_xs:
                        if not cluster or abs(xs - cluster[-1]) <= self.merge_thresh:
                            cluster.append(xs)
                        else:
                            merged_xs.append(int(np.mean(cluster)))
                            cluster = [xs]
                    if cluster:
                        merged_xs.append(int(np.mean(cluster)))

                # Visualize
                merged_line_img = piano.copy()
                xs_to_use = self.locked_merged_xs if self.locked and self.locked_merged_xs else merged_xs
                for xs in xs_to_use:
                    cv2.line(merged_line_img, (xs, 0), (xs, merged_line_img.shape[0]), (0, 255, 0), 2)

                cv2.imshow("Piano lines", merged_line_img)
                
            cropped_copy = piano_bgr.copy()
            y_pos_white = int(cropped_copy.shape[0] * 0.9)
            y_pos_black = int(cropped_copy.shape[0] * 0.7)

            cropped_pil = Image.fromarray(cv2.cvtColor(cropped_copy, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(cropped_pil)

            # Draw white keys
            # if len(xs_to_use) - 1 == len(self.notes):
            #     for i in range(len(xs_to_use) - 1):
            #         mid_x = int((xs_to_use[i] + xs_to_use[i + 1]) / 2)
            #         self.draw_note_circle(draw, mid_x, y_pos_white, self.notes[i])
            if len(xs_to_use) - 1 == len(self.notes):
                for i in range(len(xs_to_use) - 1):
                    note = self.notes[i]
                    mid_x = int((xs_to_use[i] + xs_to_use[i + 1]) / 2)
                    is_target = (
                        self.locked and isinstance(self.current_note, dict) and
                        (note in self.current_note.get("treble", []) or note in self.current_note.get("bass", []))
                    )
                    self.draw_note_circle(draw, mid_x, y_pos_white, note, is_target)

            # Draw black keys
            # for idx, note in zip(self.black_indices, self.black_notes):
            #     if idx < len(xs_to_use) - 1:
            #         mid_x = int(xs_to_use[idx])
            #         self.draw_note_circle(draw, mid_x, y_pos_black, note)
            for idx, note in zip(self.black_indices, self.black_notes):
                if idx < len(xs_to_use) - 1:
                    mid_x = int(xs_to_use[idx])
                    is_target = (
                        self.locked and isinstance(self.current_note, dict) and
                        (note in self.current_note.get("treble", []) or note in self.current_note.get("bass", []))
                    )
                    self.draw_note_circle(draw, mid_x, y_pos_black, note, is_target)

            # Convert and show
            frame_show = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
            # label_text = f"Now play: {self.current_note}" if self.current_note else "Waiting..."
            # cv2.putText(frame_show, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

            cv2.imshow("Piano Trainer", frame_show)

            key = cv2.waitKey(1) & 0xFF
            if  key == ord('l'):
                if 'x2' in locals() and merged_xs:  # Only lock if the crop exists
                    self.locked = not self.locked
                    if self.locked:
                        self.locked_merged_xs = merged_xs.copy()
                        self.locked_crop_coords = (x, y, w, h, x2, y2, w2, h2)
                        print("Locked piano layout.")
                    else:
                        self.locked_merged_xs = None
                        self.locked_crop_coords = None
                        print("Unlocked piano layout.")
                else:
                    print("Lock skipped — crop not ready.")

            if key == 27:  # ESC to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

