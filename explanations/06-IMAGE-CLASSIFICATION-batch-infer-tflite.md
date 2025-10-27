# Image Classification - batch_infer_images_tflite.py

**Location**: `image-classification/batch_infer_images_tflite.py`

## Purpose

Performs **batch image classification** on saved images using TensorFlow Lite models. Instead of processing live camera feed, this script analyzes multiple image files from a directory or a single image file, showing classification results with inference time for each.

## Batch vs Real-Time Processing

```
BATCH PROCESSING (This Script)          REAL-TIME PROCESSING
┌──────────────────────┐               ┌──────────────────────┐
│ Image 1: ESP32 95%   │               │  [Live Camera Feed]  │
│ Time: 23.5ms         │               │                      │
├──────────────────────┤               │  Current: STM32 87%  │
│ Image 2: STM32 89%   │               │  FPS: 45             │
│ Time: 24.1ms         │               │                      │
├──────────────────────┤               └──────────────────────┘
│ Image 3: Jetson 92%  │
│ Time: 23.8ms         │                   Updates every frame
└──────────────────────┘                  No permanent record

Processes all at once
Results saved/printed
```

## Use Cases

### When to use batch processing:
- **Quality control**: Analyze photos from production line
- **Dataset validation**: Check if model works on test images
- **Offline analysis**: No camera needed
- **Documentation**: Generate reports with results
- **Performance testing**: Measure average inference time

### When to use real-time:
- **Demonstrations**: Show live classification
- **Interactive applications**: Respond to user actions
- **Real-time systems**: Immediate feedback needed

## How It Works

### Workflow

```
1. Load TFLite Model
   ↓
2. Load Labels
   ↓
3. Find Images
   ├─ Single image: process one
   └─ Directory: find all .jpg/.png/.bmp
   ↓
4. For Each Image:
   ├─ Load image
   ├─ Preprocess (resize, normalize)
   ├─ Run inference
   ├─ Get probabilities
   ├─ Find top-K predictions
   └─ Print results
   ↓
5. Done
```

## Code Breakdown

### 1. Import Handling (Lines 21-36)

```python
# Try lightweight TFLite runtime first
try:
    import tflite_runtime.interpreter as tflite_rt
    Interpreter = tflite_rt.Interpreter
    print("Using tflite_runtime.Interpreter")
except ImportError:
    # Fall back to full TensorFlow
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter (via tf.lite)")
    except (ImportError, AttributeError) as e:
        raise RuntimeError("Install tensorflow or tflite-runtime") from e
```

**Package comparison:**

| Package | Size | Install Time | Use Case |
|---------|------|--------------|----------|
| tflite_runtime | ~2MB | Seconds | Production deployment |
| tensorflow | ~500MB | Minutes | Development, GPU support |

### 2. Helper Functions

#### Load Labels (Lines 39-41)
```python
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]
```

**Example labels.txt:**
```
ESP32
STM32
Jetson Nano
Arduino
Raspberry Pi
```

**Result:**
```python
labels = ["ESP32", "STM32", "Jetson Nano", "Arduino", "Raspberry Pi"]
```

#### Make Interpreter (Lines 44-47)
```python
def make_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter
```

### 3. Preprocessing (Lines 50-55)

```python
def preprocess_image_pil(img: Image.Image, input_size):
    # Convert to RGB (handles RGBA, grayscale, etc.)
    img = img.convert("RGB")

    # Resize to model's expected size
    img = img.resize((input_size[1], input_size[2]), Image.BILINEAR)

    # Convert to NumPy array
    arr = np.asarray(img).astype(np.float32)  # Shape: (H, W, 3)
    return arr
```

**Why each step:**

```
1. Convert to RGB:
   - Some images are RGBA (with transparency)
   - Some are grayscale
   - Model expects RGB

2. Resize:
   - Model trained on specific size (e.g., 320x320)
   - Images come in various sizes
   - BILINEAR interpolation for quality

3. Convert to float32:
   - Original: uint8 (0-255)
   - TFLite expects float32 (0-255, will normalize later)
```

### 4. Set Input Tensor (Lines 58-82)

```python
def set_input_tensor(interpreter, image_array):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_dtype   = input_details["dtype"]
    quant        = input_details.get("quantization", (0.0, 0))

    # image_array expected HxWx3 (uint8 0..255)
    if input_dtype == np.float32:
        # Float model: normalize 0-255 → 0-1
        inp = image_array.astype(np.float32) / 255.0
    else:
        # Quantized model: apply quantization parameters
        scale, zero_point = quant
        if scale and zero_point is not None:
            inp = image_array.astype(np.float32) / scale + zero_point
            inp = np.round(inp).astype(input_dtype)
        else:
            inp = image_array.astype(input_dtype)

    # Add batch dimension: (H,W,3) → (1,H,W,3)
    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(tensor_index, inp)
```

**Two normalization strategies:**

#### Float32 Model:
```python
# Input: [0, 255] uint8
# Process: / 255.0
# Output: [0.0, 1.0] float32

Example:
  0   → 0.0
  128 → 0.502
  255 → 1.0
```

#### Int8 Quantized Model:
```python
# Input: [0, 255] uint8
# Process: / scale + zero_point, then round
# Output: [-128, 127] int8

Example with scale=0.5, zero_point=-128:
  0   → 0/0.5 + (-128) = -128
  128 → 128/0.5 + (-128) = 128 (clipped to 127)
  255 → 255/0.5 + (-128) = 382 (clipped to 127)
```

### 5. Get Output Probabilities (Lines 85-105)

```python
def get_output_probs(interpreter):
    output_details = interpreter.get_output_details()[0]
    output_data    = interpreter.get_tensor(output_details["index"])
    out_dtype      = output_details["dtype"]
    scale, zero_point = output_details.get("quantization", (0.0, 0))

    # Remove batch dimension: (1, num_classes) → (num_classes,)
    scores = np.squeeze(output_data)

    # Dequantize if quantized output
    if out_dtype in (np.uint8, np.int8) and scale:
        scores = scale * (scores.astype(np.float32) - zero_point)

    # Apply softmax if not already probabilities
    if scores.min() < 0 or scores.max() > 1 or not np.isclose(scores.sum(), 1.0):
        exps = np.exp(scores - np.max(scores))
        probs = exps / np.sum(exps)
    else:
        probs = scores.astype(np.float32)

    return probs
```

**Dequantization formula:**
```
float_value = scale * (quantized_value - zero_point)

Example with scale=0.01, zero_point=0:
  Int8 output: [120, 85, 32, 15, 8]
  Float scores: [1.2, 0.85, 0.32, 0.15, 0.08]
  After softmax: [0.66, 0.23, 0.07, 0.03, 0.01]  ← Probabilities
```

### 6. Classification Function (Lines 108-119)

```python
def classify_image(interpreter, pil_img, labels, top_k=3):
    input_details = interpreter.get_input_details()[0]
    input_shape   = input_details["shape"]

    # Preprocess
    image_array   = preprocess_image_pil(pil_img, input_shape)

    # Set input and run inference
    set_input_tensor(interpreter, image_array)
    start = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start) * 1000.0  # Convert to ms

    # Get results
    probs = get_output_probs(interpreter)
    top_k_idx = np.argsort(probs)[-top_k:][::-1]  # Get top K indices
    results = [(labels[i], float(probs[i])) for i in top_k_idx]

    return results, inference_time
```

**Top-K selection explained:**

```python
# Example probabilities for 5 classes
probs = [0.05, 0.23, 0.02, 0.66, 0.04]

# argsort() returns indices sorted by value (ascending)
np.argsort(probs)  # [2, 4, 0, 1, 3]
                   #  ↑  ↑  ↑  ↑  ↑
                   # 0.02 → 0.66

# Get last 3 (top 3)
[-top_k:]  # [0, 1, 3]

# Reverse to descending order
[::-1]  # [3, 1, 0]

# Map to (label, probability)
# [3] → "Jetson": 0.66
# [1] → "STM32": 0.23
# [0] → "ESP32": 0.05
```

### 7. Find Images (Lines 122-124)

```python
def find_images_in_dir(d):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in Path(d).iterdir() if p.suffix.lower() in exts]
```

**Supported formats:**
- `.jpg`, `.jpeg`: JPEG compressed images
- `.png`: Lossless compressed images
- `.bmp`: Uncompressed bitmap images

### 8. Main Function (Lines 127-165)

```python
def main():
    # Parse arguments
    p = argparse.ArgumentParser()
    p.add_argument("--model",     "-m", required=True)
    p.add_argument("--labels",    "-l", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--images_dir", help="Directory with images")
    grp.add_argument("--image",      help="Single image file")
    p.add_argument("--top_k",      type=int, default=3)
    args = p.parse_args()

    # Load model and labels
    interpreter = make_interpreter(Path(args.model))
    labels = load_labels(args.labels)

    # Get images
    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(find_images_in_dir(Path(args.images_dir)))

    # Process each image
    for img_path in images:
        img = Image.open(img_path)
        results, ms = classify_image(interpreter, img, labels, top_k=args.top_k)

        print(f"\n{img_path.name} — inference {ms:.1f} ms")
        for label, prob in results:
            print(f"  {label}: {prob:.4f}")
```

## Command-Line Arguments

### Required (pick one input method):

```bash
# Option 1: Single image
--image path/to/image.jpg

# Option 2: Directory of images
--images_dir path/to/directory/
```

### Required (both):

```bash
--model   (-m)  # Path to .tflite or .lite model
--labels  (-l)  # Path to labels.txt
```

### Optional:

```bash
--top_k   # Number of predictions to show (default: 3)
```

## Usage Examples

### Example 1: Single Image
```bash
python batch_infer_images_tflite.py \
  --model models/classifier.lite \
  --labels labels.txt \
  --image test_images/esp32_board.jpg \
  --top_k 3
```

**Output:**
```
Model input details: {'name': 'serving_default_conv2d_input:0',
                       'shape': array([  1, 320, 320,   3]), 'dtype': <class 'numpy.float32'>}
Starting inference...

esp32_board.jpg — inference 24.3 ms
  ESP32: 0.8852
  ESP8266: 0.0723
  Arduino: 0.0312
```

### Example 2: Directory of Images
```bash
python batch_infer_images_tflite.py \
  --model models/classifier-int8.lite \
  --labels labels.txt \
  --images_dir ./sample-directory \
  --top_k 1
```

**Output:**
```
Model input details: {...}
Starting inference...

board_001.jpg — inference 12.8 ms
  STM32: 0.9234

board_002.jpg — inference 13.1 ms
  Jetson: 0.8876

board_003.jpg — inference 12.5 ms
  ESP32: 0.9512

... (continues for all images)
```

### Example 3: Top-5 Predictions
```bash
python batch_infer_images_tflite.py \
  --model models/classifier.lite \
  --labels labels.txt \
  --image uncertain_board.jpg \
  --top_k 5
```

**Output:**
```
uncertain_board.jpg — inference 25.1 ms
  ESP32: 0.4523      ← Uncertain prediction
  STM32: 0.3821      ← Close second
  Arduino: 0.1234
  Jetson: 0.0312
  Raspberry Pi: 0.0110
```

## Interpreting Results

### High Confidence (Good)
```
esp32_clear.jpg — inference 23.5 ms
  ESP32: 0.9823    ← 98.23% confident ✅
  STM32: 0.0123    ← Much lower alternatives
  Jetson: 0.0054
```

**What this means:**
- Model is very sure
- Image is clear and similar to training data
- Prediction is likely correct

### Low Confidence (Uncertain)
```
blurry_board.jpg — inference 24.1 ms
  ESP32: 0.4512    ← Only 45% confident ⚠️
  STM32: 0.3987    ← Close second
  Arduino: 0.1234
```

**What this means:**
- Model is uncertain
- Image might be blurry, dark, or at odd angle
- Could be object not in training set
- Consider top-2 or top-3 predictions

### Wrong Prediction Example
```
arduino_uno.jpg — inference 22.8 ms
  STM32: 0.6234    ← Wrong! (if it's actually Arduino)
  Arduino: 0.3123  ← Correct label is second
  ESP32: 0.0543
```

**Possible reasons:**
- Similar appearance between boards
- Limited training data for Arduino
- Specific angle/lighting not in training set

## Performance Analysis

### Inference Time Breakdown

```
Total time per image: ~25ms

Breakdown:
├─ Load image from disk:   5ms
├─ Preprocessing:          3ms
│  ├─ RGB conversion:      1ms
│  ├─ Resize:              1.5ms
│  └─ Array conversion:    0.5ms
├─ TFLite inference:       15ms  ← Main computation
└─ Postprocessing:         2ms
   ├─ Softmax:             1ms
   └─ Top-K selection:     1ms
```

### Speed Comparison

| Model Type | Inference Time | Speedup |
|------------|----------------|---------|
| Keras .h5 (Float32) | 45-60ms | 1x |
| TFLite Float32 | 20-30ms | 2x |
| TFLite Int8 Quantized | 10-15ms | 4x |

### Batch Performance

Processing 100 images:
```
With loading: 100 images × 25ms = 2.5 seconds
Pure inference: 100 images × 15ms = 1.5 seconds
Throughput: ~66 images/second
```

## Common Issues and Solutions

### Issue: "Model not found"
```bash
FileNotFoundError: Model not found: models/classifier.lite
```

**Solution:**
```bash
# List available models
ls -lh models/

# Use correct path
--model models/your-actual-model-name.lite
```

### Issue: "No images found"
```bash
FileNotFoundError: No images found in ./sample-directory
```

**Solution:**
```bash
# Check directory exists and has images
ls -lh sample-directory/

# Verify file extensions (.jpg, .png, .bmp)
find sample-directory/ -name "*.jpg"
```

### Issue: All predictions are wrong
**Possible causes:**
1. Model not trained on these objects
2. Different lighting/angles than training data
3. Wrong model loaded

**Debug steps:**
```bash
# 1. Verify labels match model
cat labels.txt

# 2. Test on known good image from training set
--image training_data/verified_esp32.jpg

# 3. Check top-5 to see if correct label appears
--top_k 5
```

### Issue: Slow inference
**Solutions:**
```bash
# 1. Use quantized model
--model models/classifier-int8-quantized.lite

# 2. Reduce image size before processing
# (resize images to 320x320 beforehand)

# 3. Use GPU acceleration if available
# (requires GPU delegate)
```

## Advanced Usage

### Generate CSV Report
```bash
# Run and save to file
python batch_infer_images_tflite.py \
  --model models/classifier.lite \
  --labels labels.txt \
  --images_dir ./test_images/ \
  --top_k 1 > results.csv
```

### Integration into Scripts
```python
from batch_infer_images_tflite import classify_image, make_interpreter, load_labels
from PIL import Image

# Setup
interpreter = make_interpreter("model.lite")
labels = load_labels("labels.txt")

# Classify
img = Image.open("test.jpg")
predictions, time_ms = classify_image(interpreter, img, labels, top_k=3)

# Process results
for label, prob in predictions:
    if prob > 0.8:
        print(f"High confidence: {label} ({prob:.1%})")
```

## Related Files

- `camera_infer_tflite.py`: Real-time classification from camera
- `batch_infer_h5.py`: Batch processing with Keras .h5 models
- `camera_infer_h5.py`: Real-time with unoptimized models
- `labels.txt`: Class names used by model
