# 🤟 Real-Time Sign Language Recognition

Real-time recognition of dynamic sign-language gestures from a webcam, using
**MediaPipe** hand-landmark tracking and an **LSTM** sequence model trained in
TensorFlow. Deployed as a live, in-browser web app with Streamlit.

> **Live demo:** _add your Streamlit Cloud URL here once deployed_

![demo](demo.gif) <!-- record a short clip of the app working and save it as demo.gif -->

## How it works

```
Webcam ─▶ MediaPipe HandLandmarker ─▶ 126 keypoints/frame
       ─▶ rolling 30-frame buffer ─▶ LSTM classifier ─▶ predicted sign
```

Each frame yields 21 hand landmarks × 3 coords × 2 hands = **126 features**.
A sliding window of **30 frames** is fed to a 3-layer LSTM that classifies the
motion into one of the trained signs.

## Recognized signs

`hello` · `thanks` · `yes` · `no` · `please`

## Project structure

| File | Purpose |
|------|---------|
| `collect_data.py` | Record landmark sequences per sign (with live inspection) |
| `inspect_data.py` | Replay recorded sequences to verify data quality |
| `train.py` | Train the LSTM and export `sign_model.keras` |
| `inference.py` | Local real-time recognition (OpenCV window) |
| `app.py` | Streamlit web app (browser webcam via WebRTC) |

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt streamlit-webrtc av
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (make sure `sign_model.keras` and
   `models/hand_landmarker.task` are committed — they're needed at runtime).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file** to `app.py`, and under **Advanced
   settings** choose **Python 3.11**.
4. Deploy. `requirements.txt` and `packages.txt` are picked up automatically.

## Tech stack

Python · TensorFlow · MediaPipe Tasks · OpenCV · Streamlit · streamlit-webrtc
