import os
import platform
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "gesture_cnn.keras")
MP_MODEL = os.path.join(BASE_DIR, "..", "hand_landmarker.task")

CLASS_IDS = ["good_luck", "i_love_you", "i_want_to_talk_to_you", "victory"]
DISPLAY_LABELS = {
    "good_luck": "Good luck",
    "i_love_you": "I love you",
    "i_want_to_talk_to_you": "I want to talk to you",
    "victory": "Victory",
}

IMG_SIZE = 128
PADDING = 30
SMOOTH_N = 7
MIN_CONFIDENCE = 0.75
MIN_MARGIN = 0.12

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

keras_model = tf.keras.models.load_model(MODEL_PATH)

base_options = mp_tasks.BaseOptions(
    model_asset_path=MP_MODEL,
    delegate=mp_tasks.BaseOptions.Delegate.CPU,
)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.VIDEO,
)
landmarker = vision.HandLandmarker.create_from_options(hand_options)


def open_camera():
    system = platform.system()
    attempts = []

    if system == "Darwin":
        attempts = [
            (0, cv2.CAP_AVFOUNDATION),
            (1, cv2.CAP_AVFOUNDATION),
            (0, cv2.CAP_ANY),
        ]
    else:
        attempts = [
            (0, cv2.CAP_ANY),
            (1, cv2.CAP_ANY),
        ]

    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"Camera opened on index {index}")
            return cap
        cap.release()

    if system == "Darwin":
        backend_note = "System Settings > Privacy & Security > Camera"
    else:
        backend_note = "your OS camera permission settings"

    raise SystemExit(
        "Could not open the webcam.\n"
        "Most common fixes:\n"
        f"1. Allow camera access for the app that launches Python in {backend_note}.\n"
        "2. Close Zoom/Meet/Photo Booth/another app that may already be using the camera.\n"
        "3. Try a different camera index such as 1 if you use an external webcam."
    )


def square_bbox(points, frame_w, frame_h, padding):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    box_w = max_x - min_x
    box_h = max_y - min_y
    size = max(box_w, box_h) + 2 * padding
    cx = (min_x + max_x) // 2
    cy = (min_y + max_y) // 2

    half = max(size // 2, 1)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(frame_w, cx + half)
    y2 = min(frame_h, cy + half)

    return x1, y1, x2, y2


def overlay_status(frame, label, confidence, fps):
    h, w = frame.shape[:2]

    if label == "No hand detected":
        color = (70, 70, 230)
        text = label
    elif label == "Uncertain":
        color = (0, 200, 255)
        text = f"{label}  {confidence * 100:.1f}%"
    else:
        color = (100, 220, 100)
        text = f"{label}  {confidence * 100:.1f}%"

    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        1,
    )


cap = open_camera()
pred_buffer = deque(maxlen=SMOOTH_N)
prev_time = time.time()

print("Press Q to quit")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    raw_frame = frame.copy()
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.perf_counter() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    label = "No hand detected"
    confidence = 0.0

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for p in pts:
            cv2.circle(frame, p, 6, (255, 255, 255), -1)
            cv2.circle(frame, p, 3, (0, 255, 255), -1)

        x1, y1, x2, y2 = square_bbox(pts, w, h, PADDING)
        crop = raw_frame[y1:y2, x1:x2]

        if crop.size > 0:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_r = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE))
            inp = preprocess_input(crop_r.astype("float32"))
            inp = np.expand_dims(inp, 0)
            probs = keras_model.predict(inp, verbose=0)[0]
            pred_buffer.append(probs)

            avg_probs = np.mean(pred_buffer, axis=0)
            top_indices = np.argsort(avg_probs)[::-1]
            best_idx = int(top_indices[0])
            second_idx = int(top_indices[1])
            confidence = float(avg_probs[best_idx])
            margin = confidence - float(avg_probs[second_idx])

            if confidence >= MIN_CONFIDENCE and margin >= MIN_MARGIN:
                label = DISPLAY_LABELS[CLASS_IDS[best_idx]]
            else:
                label = "Uncertain"

            box_color = (100, 220, 100) if label != "Uncertain" else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    else:
        pred_buffer.clear()

    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    prev_time = now

    overlay_status(frame, label, confidence, fps)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
