import numpy as np
import os
import cv2

# ── Config (keep in sync with collect_data.py) ────────────────────────────────
SIGNS    = ['hello', 'thanks', 'yes', 'no', 'please']
DATA_DIR = 'data'
CANVAS   = 500
# ─────────────────────────────────────────────────────────────────────────────

# Standard 21-point hand skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm
]


def draw_hand(canvas, coords63, color):
    """coords63: flat array of 21 landmarks * (x, y, z), normalized [0,1]."""
    if not coords63.any():
        return
    pts = coords63.reshape(21, 3)
    px = [(int(x * CANVAS), int(y * CANVAS)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, px[a], px[b], color, 2)
    for p in px:
        cv2.circle(canvas, p, 4, color, -1)


def report():
    print('Detection check (sequences with missing hands):\n')
    for sign in SIGNS:
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(sign_dir):
            continue
        for seq_file in sorted(os.listdir(sign_dir)):
            seq = np.load(os.path.join(sign_dir, seq_file))   # (30, 126)
            empty_frames = sum(1 for frame in seq if not frame.any())
            if empty_frames > 0:
                print(f'  {sign}/{seq_file}: {empty_frames}/{len(seq)} frames with NO hand detected')
    print('\n(sequences not listed above had a hand detected in every frame)\n')


def replay(sign, seq_index):
    path = os.path.join(DATA_DIR, sign, f'{seq_index}.npy')
    seq = np.load(path)  # (30, 126)
    print(f'Replaying {path}  — press any key to advance, q to quit')
    for frame in seq:
        canvas = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        draw_hand(canvas, frame[:63], (0, 255, 0))    # left hand  - green
        draw_hand(canvas, frame[63:], (0, 0, 255))    # right hand - red
        cv2.putText(canvas, sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Replay', canvas)
        if cv2.waitKey(150) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    report()

    # Replay specific sequences here, e.g.:
    replay('hello', 0)
    replay('thanks', 0)
