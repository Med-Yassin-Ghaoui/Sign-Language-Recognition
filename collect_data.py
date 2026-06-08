import cv2
import numpy as np
import mediapipe as mp
import os
import time
import shutil

# ── Config ────────────────────────────────────────────────────────────────────
SIGNS           = ['hello', 'thanks', 'yes', 'no', 'please']
NUM_SEQUENCES   = 30
SEQUENCE_LENGTH = 30
DATA_DIR        = 'data'
MODEL_PATH      = 'models/hand_landmarker.task'
# ─────────────────────────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


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


def draw_hand(canvas, coords63, color):
    if not coords63.any():
        return
    pts = coords63.reshape(21, 3)
    px = [(int(x * 400), int(y * 400)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 2)
    for p in px:
        cv2.circle(canvas, p, 4, color, -1)


def collect_sign(sign, landmarker, cap, start_time):
    sign_dir = os.path.join(DATA_DIR, sign)
    os.makedirs(sign_dir, exist_ok=True)

    # Wait for spacebar
    while True:
        _, frame = cap.read()
        cv2.putText(frame, f'Ready for: "{sign}"', (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, 'SPACE = start   Q = quit', (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow('Data Collection', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            return 'quit'

    for seq in range(NUM_SEQUENCES):
        for countdown in range(3, 0, -1):
            _, frame = cap.read()
            cv2.putText(frame, f'"{sign}"  Seq {seq+1}/{NUM_SEQUENCES}  Starting in {countdown}',
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1000)

        frames = []
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                break
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = int((time.time() - start_time) * 1000)
            result    = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
            frames.append(extract_keypoints(result))

            cv2.putText(frame, f'Recording "{sign}"  {seq+1}/{NUM_SEQUENCES}  frame {frame_num+1}',
                        (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return 'quit'

        np.save(os.path.join(sign_dir, str(seq)), np.array(frames))

    print(f'\nDone recording "{sign}". Replaying for inspection...')
    return 'collected'


def inspect_sign(sign):
    sign_dir = os.path.join(DATA_DIR, sign)
    files    = sorted(os.listdir(sign_dir))
    print(f'Showing {len(files)} sequences. Any key = next frame, Q = stop replay.')

    for seq_file in files:
        seq     = np.load(os.path.join(sign_dir, seq_file))
        empty   = sum(1 for f in seq if not f.any())
        stopped = False

        for frame in seq:
            canvas = np.zeros((400, 400, 3), dtype=np.uint8)
            draw_hand(canvas, frame[:63], (0, 255, 0))
            draw_hand(canvas, frame[63:], (0, 0, 255))
            cv2.putText(canvas, f'{sign} / {seq_file}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if empty:
                cv2.putText(canvas, f'{empty} empty frames!', (10, 380),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow('Inspection', canvas)
            if cv2.waitKey(120) & 0xFF == ord('q'):
                stopped = True
                break

        if stopped:
            break

    cv2.destroyWindow('Inspection')


def show_menu(cap):
    status = {sign: os.path.isdir(os.path.join(DATA_DIR, sign)) and
              bool(os.listdir(os.path.join(DATA_DIR, sign))) for sign in SIGNS}

    while True:
        _, frame = cap.read()
        y = 40
        cv2.putText(frame, 'Select a sign to record:', (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, sign in enumerate(SIGNS):
            done  = status[sign]
            color = (0, 255, 0) if done else (0, 200, 255)
            tag   = '[done]' if done else '[empty]'
            cv2.putText(frame, f'  {i+1} = {sign}  {tag}', (15, y + 35 * (i + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, '  Q = quit', (15, y + 35 * (len(SIGNS) + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
        cv2.imshow('Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        for i, sign in enumerate(SIGNS):
            if key == ord(str(i + 1)):
                return sign


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap        = cv2.VideoCapture(0)
start_time = time.time()

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        sign = show_menu(cap)
        if sign is None:
            break

        # If already has data, ask to redo or inspect
        sign_dir = os.path.join(DATA_DIR, sign)
        if os.path.isdir(sign_dir) and os.listdir(sign_dir):
            while True:
                _, frame = cap.read()
                cv2.putText(frame, f'"{sign}" already has data.', (15, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(frame, 'R = re-record   I = inspect   M = back to menu', (15, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow('Data Collection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    shutil.rmtree(sign_dir)
                    break
                if key == ord('i'):
                    inspect_sign(sign)
                    break
                if key == ord('m'):
                    sign = None
                    break
            if sign is None:
                continue

        result = collect_sign(sign, landmarker, cap, start_time)
        if result == 'quit':
            break

        inspect_sign(sign)

        # After inspection: keep or redo
        while True:
            _, frame = cap.read()
            cv2.putText(frame, f'Keep "{sign}" recordings?', (15, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, 'Y = keep & back to menu   R = redo   Q = quit', (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow('Data Collection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                print(f'"{sign}" saved.')
                break
            if key == ord('r'):
                shutil.rmtree(sign_dir)
                result = collect_sign(sign, landmarker, cap, start_time)
                if result == 'quit':
                    break
                inspect_sign(sign)
            if key == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
print('Done.')
