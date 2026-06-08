import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
from collections import deque

# ── Config (keep in sync with collect_data.py) ────────────────────────────────
SIGNS                = ['hello', 'thanks', 'yes', 'no', 'please']
SEQUENCE_LENGTH      = 30
CONFIDENCE_THRESHOLD = 0.85
MODEL_PATH           = 'models/hand_landmarker.task'
MODEL_FILE           = 'sign_model.keras'
# ─────────────────────────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode


def extract_keypoints(result):
    lh = np.zeros(63)
    rh = np.zeros(63)
    for i, handedness in enumerate(result.handedness):
        label = handedness[0].category_name
        coords = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[i]]).flatten()
        if label == 'Left':
            lh = coords
        else:
            rh = coords
    return np.concatenate([lh, rh])


def draw_bounding_boxes(frame, result):
    h, w = frame.shape[:2]
    for i, handedness in enumerate(result.handedness):
        landmarks = result.hand_landmarks[i]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1, y1 = int(min(xs) * w) - 20, int(min(ys) * h) - 20
        x2, y2 = int(max(xs) * w) + 20, int(max(ys) * h) + 20
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        label = handedness[0].category_name
        color = (0, 255, 0) if label == 'Left' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


model    = load_model(MODEL_FILE)
sequence = deque(maxlen=SEQUENCE_LENGTH)
label    = ''

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)
start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp  = int((time.time() - start_time) * 1000)
        result     = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
        sequence.append(extract_keypoints(result))

        if len(sequence) == SEQUENCE_LENGTH:
            probs = model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
            if probs.max() > CONFIDENCE_THRESHOLD:
                label = SIGNS[np.argmax(probs)]

        draw_bounding_boxes(frame, result)
        cv2.putText(frame, label, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
