#!/usr/bin/env python3
"""
camera_infer_tflite.py
Run a TFLite image-classification model on live camera frames and display predictions.

Usage example:
  python camera_infer_tflite.py --model float32.lite --labels labels.txt --camera 0 --top_k 1 --width 320 --height 320

Changes:
 - Simplified import: attempt tflite_runtime first; if unavailable, fall back to tf.lite.Interpreter.
 - Prints model input details for debug.
 - Preprocesses frames, sets tensor, invokes, then overlays predictions and FPS.
"""

import argparse
import time
import sys
import os

import cv2
import numpy as np
from PIL import Image

# Import the interpreter class: try tflite_runtime, else fallback
try:
    import tflite_runtime.interpreter as tflite_rt
    Interpreter = tflite_rt.Interpreter
    print("Using tflite_runtime.Interpreter")
except ImportError:
    try:
        # Use TensorFlow's built-in TFLite interpreter
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter")
    except ImportError as e:
        raise RuntimeError(
            "Could not import tflite_runtime or tensorflow.lite.Interpreter. "
            "Install a compatible version of tensorflow or tflite-runtime."
        ) from e

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]

def make_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter

def preprocess_cv2(frame, input_shape):
    # frame: BGR HxW x3 (cv2)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    # Resize to model input W,H
    img = img.resize((input_shape[2], input_shape[1]), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return arr

def set_input_tensor(interpreter, image_array):
    input_details = interpreter.get_input_details()[0]
    tensor_index   = input_details["index"]
    input_dtype    = input_details["dtype"]
    scale, zero_point = input_details.get("quantization", (0.0, 0))

    if input_dtype == np.float32:
        # Float model: assume 0..1 normalization
        inp = image_array.astype(np.float32) / 255.0
    else:
        # Quantized model: use scale & zero_point
        if scale and zero_point is not None:
            inp = image_array.astype(np.float32) / scale + zero_point
            inp = np.round(inp).astype(input_dtype)
        else:
            inp = image_array.astype(input_dtype)

    # Add batch dimension
    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(tensor_index, inp)

def get_output_probs(interpreter):
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details["index"])
    out_dtype   = output_details["dtype"]
    scale, zero_point = output_details.get("quantization", (0.0, 0))

    scores = np.squeeze(output_data)
    if out_dtype in (np.uint8, np.int8) and scale:
        scores = scale * (scores.astype(np.float32) - zero_point)

    # If these are not already probabilities, apply softmax
    if scores.min() < 0 or scores.max() > 1 or not np.isclose(scores.sum(), 1.0):
        exps = np.exp(scores - np.max(scores))
        probs = exps / np.sum(exps)
    else:
        probs = scores.astype(np.float32)
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  "-m", required=True, help="Path to .tflite or .lite model file")
    parser.add_argument("--labels", "-l", required=True, help="Path to labels.txt (one label per line)")
    parser.add_argument("--camera", type=int, default=0,   help="camera index for cv2.VideoCapture()")
    parser.add_argument("--top_k",  type=int, default=1,   help="Show top K predictions")
    parser.add_argument("--width",  type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = make_interpreter(args.model)

    input_details = interpreter.get_input_details()[0]
    input_shape   = input_details["shape"]
    print(f"Model input shape: {input_shape}, input dtype: {input_details['dtype']}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera index", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_avg = None
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        # Preprocess frame
        img_arr = preprocess_cv2(frame, input_shape)
        set_input_tensor(interpreter, img_arr)
        interpreter.invoke()
        probs = get_output_probs(interpreter)

        top_idx = np.argsort(probs)[-args.top_k:][::-1]
        predictions = [(labels[i] if i < len(labels) else str(i), float(probs[i])) for i in top_idx]

        # Overlay predictions
        y0 = 30
        for i, (lbl, p) in enumerate(predictions):
            text = f"{lbl}: {p:.2f}"
            cv2.putText(frame, text, (10, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

        # FPS smoothing
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
        fps_avg = fps if fps_avg is None else fps_avg * 0.9 + fps * 0.1
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0),2)

        cv2.imshow("TFLite Camera Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
