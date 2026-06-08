import os
# MediaPipe and TensorFlow each bundle their own native protobuf runtime.
# On Linux they clash at import time and segfault, so force the pure-Python
# protobuf implementation to remove the duplicate native load. Must be set
# before mediapipe / tensorflow are imported.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import av
import cv2
import time
import numpy as np
import streamlit as st
import mediapipe as mp
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from tensorflow.keras.models import load_model

# ── Config (keep in sync with collect_data.py / train.py) ─────────────────────
SIGNS                = ['hello', 'thanks', 'yes', 'no', 'please']
SEQUENCE_LENGTH      = 30
CONFIDENCE_THRESHOLD = 0.85
PREDICT_EVERY        = 5     # run the LSTM only every N frames (huge fps win)
MODEL_PATH           = 'models/hand_landmarker.task'
MODEL_FILE           = 'sign_model.keras'
# ─────────────────────────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# Public STUN server so WebRTC can negotiate a connection through the browser.
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


@st.cache_resource
def get_model():
    return load_model(MODEL_FILE)


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
        x1, y1 = max(int(min(xs) * w) - 20, 0), max(int(min(ys) * h) - 20, 0)
        x2, y2 = min(int(max(xs) * w) + 20, w), min(int(max(ys) * h) + 20, h)
        label = handedness[0].category_name
        color = (0, 255, 0) if label == 'Left' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.model     = get_model()
        self.sequence  = deque(maxlen=SEQUENCE_LENGTH)
        self.label     = ''
        self.frame_idx = 0
        self.landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        timestamp = int(self.frame_idx * 33)   # ~30 fps, must be monotonic
        self.frame_idx += 1
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
        )
        self.sequence.append(extract_keypoints(result))

        # Keypoints are extracted every frame to keep the buffer continuous,
        # but the LSTM only runs every PREDICT_EVERY frames to avoid stalling.
        if len(self.sequence) == SEQUENCE_LENGTH and self.frame_idx % PREDICT_EVERY == 0:
            probs = self.model(np.expand_dims(np.array(self.sequence), axis=0), training=False).numpy()[0]
            if probs.max() > CONFIDENCE_THRESHOLD:
                self.label = f'{SIGNS[np.argmax(probs)]} ({probs.max():.0%})'

        draw_bounding_boxes(img, result)
        cv2.putText(img, self.label, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sign Language Recognition", page_icon="🤟", layout="centered")
st.title("🤟 Real-Time Sign Language Recognition")
st.write(
    "Live hand-gesture recognition with MediaPipe + an LSTM model. "
    "Allow camera access and sign one of the words below."
)
st.info("**Recognized signs:** " + ", ".join(f"`{s}`" for s in SIGNS))

webrtc_streamer(
    key="sign-language",
    video_processor_factory=SignProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 30}},
        "audio": False,
    },
    async_processing=True,
)

st.caption("Built with Streamlit, MediaPipe Tasks, and TensorFlow. Source on GitHub.")
