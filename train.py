import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# ── Config (keep in sync with collect_data.py) ────────────────────────────────
SIGNS           = ['hello', 'thanks', 'yes', 'no', 'please']
SEQUENCE_LENGTH = 30
DATA_DIR        = 'data'
MODEL_OUT       = 'sign_model.keras'
# ─────────────────────────────────────────────────────────────────────────────

# Load data
X, y = [], []
for label, sign in enumerate(SIGNS):
    sign_dir = os.path.join(DATA_DIR, sign)
    for seq_file in sorted(os.listdir(sign_dir)):
        sequence = np.load(os.path.join(sign_dir, seq_file))
        X.append(sequence)
        y.append(label)

X = np.array(X)                              # (N, 30, 126)
y = to_categorical(y, num_classes=len(SIGNS))

print(f'Dataset: {X.shape[0]} samples, {len(SIGNS)} classes')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    LSTM(64,  return_sequences=True, input_shape=(SEQUENCE_LENGTH, 126)),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(64,          activation='relu'),
    Dense(32,          activation='relu'),
    Dense(len(SIGNS),  activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {acc:.2%}')

model.save(MODEL_OUT)
print(f'Model saved to {MODEL_OUT}')
