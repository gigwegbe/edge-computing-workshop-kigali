#!/usr/bin/env python3
"""
batch_infer_h5.py
Run a Keras .h5 classification model on local images (single image or directory).

Examples:
  python batch_infer_h5.py --model model.h5 --labels labels.txt --images_dir ./images --top_k 3
  python batch_infer_h5.py --model model.h5 --labels labels.txt --image img1.jpg --norm -1_1
  python batch_infer_h5.py --model model.h5 --labels labels.txt --image img1.jpg --input_size 128 128

Notes:
  - The script will try to infer the model input size from model.input_shape.
    If height or width are None, you can pass --input_size H W to override.
  - Normalization choices: 0_1 -> scale 0..255 to 0..1 (default). -1_1 -> scale to -1..1.
"""
import argparse
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
import tensorflow as tf


def load_labels(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def infer_input_size_from_model(model) -> Tuple[int, int, int]:
    # model.input_shape usually looks like (None, H, W, C) or (None, None, None, 3)
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) != 4:
        raise RuntimeError(f"Unexpected model input shape: {shape}")
    _, h, w, c = shape
    # if any of h/w is None, return None to indicate override may be needed
    return h, w, c


def preprocess_pil(img: Image.Image, target_hw: Tuple[int, int], norm: str = "0_1") -> np.ndarray:
    # Convert to RGB and resize
    img = img.convert("RGB")
    img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # H x W x 3
    if norm == "0_1":
        arr = arr / 255.0
    elif norm == "-1_1":
        arr = (arr / 127.5) - 1.0
    else:
        raise ValueError("Unsupported norm: choose '0_1' or '-1_1'")
    return arr


def top_k_from_probs(probs: np.ndarray, labels: List[str], k: int):
    idx = np.argsort(probs)[-k:][::-1]
    return [(labels[i] if i < len(labels) else str(i), float(probs[i])) for i in idx]


def find_images_in_dir(d: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in d.iterdir() if p.suffix.lower() in exts])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", "-m", required=True, help=".h5 Keras model")
    p.add_argument("--labels", "-l", required=True, help="labels.txt (one label per line)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--images_dir", help="directory with images")
    grp.add_argument("--image", help="single image file")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--norm", choices=["0_1", "-1_1"], default="0_1",
                   help="Normalization for float models: 0_1 (default) or -1_1")
    p.add_argument("--input_size", nargs=2, type=int, metavar=("H", "W"),
                   help="Optional override input size (height width)")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    labels = load_labels(Path(args.labels))

    # Load model
    model = tf.keras.models.load_model(str(model_path))
    h, w, c = infer_input_size_from_model(model)
    if args.input_size:
        h, w = args.input_size
    else:
        # If model has None dims, pick sensible default 224x224
        if h is None or w is None:
            print("Warning: model input shape has dynamic dimensions; defaulting to 224x224. "
                  "You can pass --input_size H W to override.")
            h, w = 224, 224
    target_hw = (h, w)
    print(f"Using input size HxW: {h}x{w}, channels={c}")

    # Prepare images list
    if args.image:
        images = [Path(args.image)]
    else:
        images = find_images_in_dir(Path(args.images_dir))
        if not images:
            raise FileNotFoundError(f"No images found in {args.images_dir}")

    # Run inference
    for img_path in images:
        img = Image.open(img_path)
        x = preprocess_pil(img, target_hw, norm=args.norm)
        # add batch dim
        x_batch = np.expand_dims(x, axis=0)
        t0 = time.time()
        preds = model.predict(x_batch)
        t_ms = (time.time() - t0) * 1000.0
        preds = np.squeeze(preds)
        # If model returns logits, softmax them
        if preds.min() < 0 or not np.isclose(preds.sum(), 1.0):
            exps = np.exp(preds - np.max(preds))
            probs = exps / np.sum(exps)
        else:
            probs = preds.astype(np.float32)

        topk = top_k_from_probs(probs, labels, args.top_k)
        print(f"\n{img_path.name} — inference {t_ms:.1f} ms")
        for lbl, p in topk:
            print(f"  {lbl}: {p:.4f}")


if __name__ == "__main__":
    main()
