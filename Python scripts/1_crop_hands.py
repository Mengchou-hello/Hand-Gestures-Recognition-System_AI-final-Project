import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import shutil

INPUT_DIR  = "../dataset"
OUTPUT_DIR = "../cropped_dataset"
IMG_SIZE   = 128
PADDING    = 30
MODEL_FILE = "../hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_FILE):
    print("Downloading hand landmarker model (~25MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("Download complete.")

base_options = python.BaseOptions(
    model_asset_path=MODEL_FILE,
    delegate=python.BaseOptions.Delegate.CPU,
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
try:
    detector = vision.HandLandmarker.create_from_options(options)
except RuntimeError as exc:
    raise SystemExit(
        "MediaPipe hand detector could not start.\n"
        "If this is macOS, your current MediaPipe build may be trying to use an unsupported graphics path.\n"
        "Try the project's virtual environment and reinstall MediaPipe if the problem continues."
    ) from exc

for class_name in os.listdir(INPUT_DIR):
    in_class_dir  = os.path.join(INPUT_DIR, class_name)
    out_class_dir = os.path.join(OUTPUT_DIR, class_name)

    if not os.path.isdir(in_class_dir):
        continue
    if os.path.isdir(out_class_dir):
        shutil.rmtree(out_class_dir)
    os.makedirs(out_class_dir, exist_ok=True)

    saved, failed = 0, 0
    for fname in os.listdir(in_class_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(in_class_dir, fname)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is None:
            failed += 1
            continue

        h, w = img_bgr.shape[:2]
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result   = detector.detect(mp_image)

        if not result.hand_landmarks:
            failed += 1
            continue

        landmarks = result.hand_landmarks[0]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]

        x1 = max(0, int(min(xs)) - PADDING)
        y1 = max(0, int(min(ys)) - PADDING)
        x2 = min(w, int(max(xs)) + PADDING)
        y2 = min(h, int(max(ys)) + PADDING)

        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            failed += 1
            continue

        crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        cv2.imwrite(os.path.join(out_class_dir, fname), crop_resized)
        saved += 1

    print(f"{class_name}: saved={saved}, failed={failed}")

detector.close()
print("\nDone! Cropped dataset ready in cropped_dataset/")
