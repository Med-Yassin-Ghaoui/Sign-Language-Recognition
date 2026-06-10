# 🤟 Real-Time Sign Language Recognition

Real-time recognition of dynamic sign-language gestures from a webcam, using
**MediaPipe** hand-landmark tracking and an **LSTM** sequence model trained in
TensorFlow. Deployed as a live, in-browser web app with Streamlit.

> **[Live demo on Streamlit Cloud](https://share.streamlit.io/)**
>
> ⚠️ **Note:** The free tier app goes to sleep after 1 hour of inactivity.
> If you see a "degraded performance" message or blank page, contact the admin
> (yassine.ghaoui12@gmail.com) to reactivate it.

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
| `convert_to_tflite.py` | Convert the Keras model to `sign_model.tflite` for deployment |
| `inference.py` | Local real-time recognition (OpenCV window) |
| `app.py` | Streamlit web app (browser webcam via WebRTC) |

> **Note on the runtime:** the web app runs the model with the standalone
> **LiteRT** (`ai-edge-litert`) interpreter and never imports TensorFlow.
> TensorFlow and MediaPipe each statically link their own protobuf/abseil
> runtimes, which clash and segfault when loaded together on Linux — so the
> deployed app avoids TensorFlow entirely. TensorFlow is only needed locally
> for `train.py` and `convert_to_tflite.py`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

To retrain or re-export the model you also need TensorFlow:
`pip install tensorflow` then run `python train.py` and `python convert_to_tflite.py`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub. Make sure **`sign_model.tflite`** and
   **`models/hand_landmarker.task`** are committed — they're needed at runtime.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file** to `app.py`, and under **Advanced
   settings** choose **Python 3.11**.
4. Deploy. `requirements.txt` and `packages.txt` are picked up automatically.

## Tech stack

Python · TensorFlow (training) · TFLite/LiteRT (inference) · MediaPipe Tasks · OpenCV · Streamlit · streamlit-webrtc
