#!/usr/bin/env python3
"""
camera_infer_h5.py
Run a Keras .h5 model on live camera frames and display predictions.

Example:
  python camera_infer_h5.py --model model.h5 --labels labels.txt --camera 0 --top_k 1

Notes:
  - Use --input_size H W to override a model that has dynamic input dims.
  - Normalization: default is 0_1 (0..1). Use --norm -1_1 if model expects -1..1.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def load_labels(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def infer_input_size_from_model(model) -> Tuple[int, int, int]:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected model input shape: {shape}")
    _, h, w, c = shape
    return h, w, c


def preprocess_frame(frame: np.ndarray, target_hw: Tuple[int, int], norm: str = "0_1") -> np.ndarray:
    # frame is BGR HxW x3
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    arr = np.asarray(pil).astype(np.float32)
    if norm == "0_1":
        arr = arr / 255.0
    elif norm == "-1_1":
        arr = (arr / 127.5) - 1.0
    else:
        raise ValueError("Unsupported norm")
    return arr


def top_k_from_probs(probs: np.ndarray, labels: List[str], k: int):
    idx = np.argsort(probs)[-k:][::-1]
    return [(labels[i] if i < len(labels) else str(i), float(probs[i])) for i in idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True)
    p.add_argument("--labels", "-l", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--norm", choices=["0_1", "-1_1"], default="0_1")
    p.add_argument("--input_size", nargs=2, type=int, metavar=("H", "W"),
                   help="Optional override input size (height width)")
    args = p.parse_args()

    labels = load_labels(Path(args.labels))
    model = tf.keras.models.load_model(str(Path(args.model)))
    h, w, c = infer_input_size_from_model(model)
    if args.input_size:
        h, w = args.input_size
    else:
        if h is None or w is None:
            print("Warning: model input shape has dynamic dims; defaulting to 224x224. Use --input_size to override.")
            h, w = 224, 224
    target_hw = (h, w)
    print(f"Using input size HxW: {h}x{w}, channels={c}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera", args.camera)
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_avg = None
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera")
            break

        x = preprocess_frame(frame, target_hw, norm=args.norm)
        x_batch = np.expand_dims(x, axis=0)
        t_infer0 = time.time()
        preds = model.predict(x_batch)
        t_infer = (time.time() - t_infer0) * 1000.0
        preds = np.squeeze(preds)
        if preds.min() < 0 or not np.isclose(preds.sum(), 1.0):
            exps = np.exp(preds - np.max(preds))
            probs = exps / np.sum(exps)
        else:
            probs = preds.astype(np.float32)
        top = top_k_from_probs(probs, labels, args.top_k)

        # Overlay predictions
        y0 = 30
        for i, (lbl, p) in enumerate(top):
            text = f"{lbl}: {p:.2f}"
            cv2.putText(frame, text, (10, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, f"Infer: {t_infer:.0f}ms", (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        # FPS smoothing
        t1 = time.time()
        fps = 1.0 / (t1 - t0) if (t1 - t0) > 0 else 0.0
        fps_avg = fps if fps_avg is None else fps_avg * 0.9 + fps * 0.1
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)

        cv2.imshow("Keras .h5 Camera Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
