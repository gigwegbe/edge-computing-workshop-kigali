#!/usr/bin/env python3
"""
batch_infer_images_tflite.py
Run a TFLite (.tflite or .lite) image classification model on local images.

Usage:
  python batch_infer_images_tflite.py --model float32.lite --labels labels.txt --images_dir ./images --top_k 3
  python batch_infer_images_tflite.py --model int8-quantized.lite --labels labels.txt --image image1.jpg

The script auto-detects float vs quantized model input/output and applies
appropriate quantize/dequantize steps.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Try tflite_runtime first (lightweight), fall back to tensorflow.lite or tf.lite
try:
    import tflite_runtime.interpreter as tflite_rt
    Interpreter = tflite_rt.Interpreter
    print("Using tflite_runtime.Interpreter")
except ImportError:
    try:
        import tensorflow as tf
        # Try tf.lite.Interpreter
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter (via tf.lite)")
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "Could not import tflite-runtime or tensorflow/ tf.lite Interpreter. "
            "Install a compatible version of tensorflow or tflite-runtime."
        ) from e


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def make_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image_pil(img: Image.Image, input_size):
    # convert to RGB
    img = img.convert("RGB")
    img = img.resize((input_size[1], input_size[2]), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # HxWx3
    return arr


def set_input_tensor(interpreter, image_array):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_dtype   = input_details["dtype"]
    quant        = input_details.get("quantization", (0.0, 0))

    # image_array expected HxWx3 (uint8 0..255)
    if input_dtype == np.float32:
        # Default normalization: map 0..255 -> 0.0..1.0
        inp = image_array.astype(np.float32) / 255.0
    else:
        # Quantized model: quantize input using scale & zero_point
        scale, zero_point = quant
        if scale and zero_point is not None:
            # assume image_array in 0..255 float
            inp = image_array.astype(np.float32) / scale + zero_point
            # round & cast to dtype
            inp = np.round(inp).astype(input_dtype)
        else:
            # No quant params: just cast
            inp = image_array.astype(input_dtype)

    # Add batch dimension
    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(tensor_index, inp)


def get_output_probs(interpreter):
    output_details = interpreter.get_output_details()[0]
    output_data    = interpreter.get_tensor(output_details["index"])
    out_dtype      = output_details["dtype"]
    scale, zero_point = output_details.get("quantization", (0.0, 0))

    # Remove batch dim
    scores = np.squeeze(output_data)

    # Dequantize if necessary
    if out_dtype in (np.uint8, np.int8) and scale:
        scores = scale * (scores.astype(np.float32) - zero_point)

    # If scores are not probabilities (e.g. logits), softmax them safely
    if scores.min() < 0 or scores.max() > 1 or not np.isclose(scores.sum(), 1.0):
        exps = np.exp(scores - np.max(scores))
        probs = exps / np.sum(exps)
    else:
        probs = scores.astype(np.float32)

    return probs


def classify_image(interpreter, pil_img, labels, top_k=3):
    input_details = interpreter.get_input_details()[0]
    input_shape   = input_details["shape"]
    image_array   = preprocess_image_pil(pil_img, input_shape)
    set_input_tensor(interpreter, image_array)
    start = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start) * 1000.0
    probs = get_output_probs(interpreter)
    top_k_idx = np.argsort(probs)[-top_k:][::-1]
    results = [(labels[i] if i < len(labels) else str(i), float(probs[i])) for i in top_k_idx]
    return results, inference_time


def find_images_in_dir(d):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in Path(d).iterdir() if p.suffix.lower() in exts]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     "-m", required=True, help="Path to .tflite or .lite model")
    p.add_argument("--labels",    "-l", required=True, help="Path to labels.txt (one label per line)")
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--images_dir", help="Directory with images to run inference on")
    grp.add_argument("--image",      help="Single image file to run inference on")
    p.add_argument("--top_k",      type=int, default=3, help="Show top K predictions")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    labels = load_labels(args.labels)

    interpreter     = make_interpreter(model_path)
    input_details   = interpreter.get_input_details()[0]
    print(f"Model input details: {input_details}")
    print("Starting inference...")

    images = []
    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(find_images_in_dir(Path(args.images_dir)))
        if not images:
            raise FileNotFoundError(f"No images found in {args.images_dir}")

    for img_path in images:
        img     = Image.open(img_path)
        results, ms = classify_image(interpreter, img, labels, top_k=args.top_k)
        print(f"\n{img_path.name} — inference {ms:.1f} ms")
        for label, prob in results:
            print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
