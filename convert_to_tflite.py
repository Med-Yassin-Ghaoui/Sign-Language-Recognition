"""One-time: convert the trained Keras LSTM to a TFLite model that runs with
ONLY builtin ops (no TensorFlow Flex delegate), so the deployed app can do
inference via the standalone ai-edge-litert interpreter without importing
TensorFlow (which segfaults next to MediaPipe on Linux).

The trick: rebuild the model with unroll=True. With a fixed 30-frame sequence,
the LSTM loop expands into plain ops that map to TFLite builtins.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

SEQUENCE_LENGTH = 30
NUM_FEATURES    = 126
NUM_CLASSES     = 5
KERAS_FILE      = 'sign_model.keras'
TFLITE_FILE     = 'sign_model.tflite'

trained = tf.keras.models.load_model(KERAS_FILE)

# Same architecture as train.py, but unrolled so it converts to builtin ops.
unrolled = Sequential([
    Input(shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    LSTM(64,  return_sequences=True, unroll=True),
    LSTM(128, return_sequences=True, unroll=True),
    LSTM(64,  unroll=True),
    Dense(64,          activation='relu'),
    Dense(32,          activation='relu'),
    Dense(NUM_CLASSES, activation='softmax'),
])
unrolled.set_weights(trained.get_weights())   # weights are identical

converter = tf.lite.TFLiteConverter.from_keras_model(unrolled)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]   # no Flex!
tflite_model = converter.convert()
with open(TFLITE_FILE, 'wb') as f:
    f.write(tflite_model)
print(f'Wrote {TFLITE_FILE} ({len(tflite_model)/1024:.0f} KB)')

# ── Verify TFLite output matches Keras on random input ────────────────────────
x = np.random.rand(1, SEQUENCE_LENGTH, NUM_FEATURES).astype(np.float32)
keras_out = trained(x, training=False).numpy()[0]

interp = tf.lite.Interpreter(model_path=TFLITE_FILE)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
interp.set_tensor(inp['index'], x)
interp.invoke()
tflite_out = interp.get_tensor(out['index'])[0]

print('Keras :', np.round(keras_out, 4))
print('TFLite:', np.round(tflite_out, 4))
print('Max abs diff:', np.max(np.abs(keras_out - tflite_out)))
print('Same prediction:', np.argmax(keras_out) == np.argmax(tflite_out))
