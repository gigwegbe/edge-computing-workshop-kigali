# Image Classification - camera_infer_tflite.py

**Location**: `image-classification/camera_infer_tflite.py`

## Purpose

Runs **image classification** on live camera feed using optimized TensorFlow Lite models. Instead of detecting WHERE objects are (like object detection), this identifies WHAT the entire image/frame represents.

## Image Classification vs Object Detection

```
IMAGE CLASSIFICATION                    OBJECT DETECTION
┌─────────────────────┐                ┌─────────────────────┐
│                     │                │  ┌────┐             │
│                     │                │  │ESP │             │
│      ESP32          │  →  "ESP32"    │  └────┘  ┌─────┐   │
│                     │                │          │STM32│   │
│    (entire image)   │                │          └─────┘   │
└─────────────────────┘                └─────────────────────┘
  Single label                          Multiple boxes + labels
  Top-K predictions                     Bounding boxes
  Confidence score                      Confidence per box
```

## Key Concepts

### Top-K Predictions
Instead of just the best prediction, show the top K most likely classes:

```
Image shown to model:
└─ [Photo of ESP32]

Top-3 Predictions (K=3):
1. ESP32:  88.5% confidence
2. ESP8266: 8.2% confidence
3. Arduino:  2.1% confidence
```

**Why useful:**
- See what the model "thinks" it might be
- Understand model uncertainty
- Debug classification errors

## How It Works

### Workflow Diagram

```
Camera → Capture Frame → Preprocess → TFLite Model → Softmax → Top-K → Display
         (640x480)        (320x320)    Inference     Probs    Results  (with FPS)
                          normalize                  sort
```

## Code Breakdown

### 1. Interpreter Import (Lines 24-39)

```python
# Try tflite_runtime first (lightweight)
try:
    import tflite_runtime.interpreter as tflite_rt
    Interpreter = tflite_rt.Interpreter
    print("Using tflite_runtime.Interpreter")
except ImportError:
    # Fall back to TensorFlow's built-in TFLite
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        print("Using tensorflow.lite.Interpreter")
    except ImportError as e:
        raise RuntimeError("Could not import interpreter") from e
```

**Why two options:**
- `tflite_runtime`: Lightweight package (~2MB), faster install
- `tensorflow.lite`: Full TensorFlow package (~500MB)
- Script tries lightweight first, falls back to full package

### 2. Load Labels (Lines 41-43)

```python
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]
```

**labels.txt format:**
```
ESP32
STM32
Jetson
Arduino
Raspberry Pi
```

**Result:**
```python
labels = ["ESP32", "STM32", "Jetson", "Arduino", "Raspberry Pi"]
labels[0]  # "ESP32"
labels[2]  # "Jetson"
```

### 3. Make Interpreter (Lines 45-48)

```python
def make_interpreter(model_path):
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    return interpreter
```

**Two-step process:**
1. Create interpreter from model file
2. Allocate memory for input/output tensors

### 4. Preprocessing (Lines 50-57)

```python
def preprocess_cv2(frame, input_shape):
    # frame: BGR HxWx3 (cv2 format)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
    img = Image.fromarray(img)                    # NumPy → PIL
    img = img.resize((input_shape[2], input_shape[1]), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)
    return arr
```

**Step-by-step transformation:**
```
Input:  BGR frame (480, 640, 3)
   ↓
1. Convert to RGB (480, 640, 3)  ← cv2 uses BGR, models expect RGB
   ↓
2. Convert to PIL Image
   ↓
3. Resize to (320, 320, 3)      ← Model input size
   ↓
4. Convert to float32 array
   ↓
Output: (320, 320, 3) float32 array
```

### 5. Set Input Tensor (Lines 59-78)

```python
def set_input_tensor(interpreter, image_array):
    input_details = interpreter.get_input_details()[0]
    tensor_index   = input_details["index"]
    input_dtype    = input_details["dtype"]
    scale, zero_point = input_details.get("quantization", (0.0, 0))

    if input_dtype == np.float32:
        # Float model: normalize 0-255 → 0-1
        inp = image_array.astype(np.float32) / 255.0
    else:
        # Quantized model: use scale & zero_point
        if scale and zero_point is not None:
            inp = image_array.astype(np.float32) / scale + zero_point
            inp = np.round(inp).astype(input_dtype)
        else:
            inp = image_array.astype(input_dtype)

    # Add batch dimension: (320,320,3) → (1,320,320,3)
    inp = np.expand_dims(inp, axis=0)
    interpreter.set_tensor(tensor_index, inp)
```

**Handles two model types:**

#### Float32 Model:
```
Input range:  0 - 255 (uint8)
Normalize:    / 255.0
Output range: 0.0 - 1.0 (float32)
```

#### Quantized (Int8) Model:
```
Input range:  0 - 255 (uint8)
Quantize:     / scale + zero_point
Output range: -128 - 127 (int8)

Example with scale=0.5, zero_point=0:
  255 → 255/0.5 + 0 = 510 → clipped to 127
  128 → 128/0.5 + 0 = 256 → clipped to 127
  0   → 0/0.5 + 0   = 0
```

### 6. Get Output Probabilities (Lines 80-96)

```python
def get_output_probs(interpreter):
    output_details = interpreter.get_output_details()[0]
    output_data = interpreter.get_tensor(output_details["index"])
    out_dtype   = output_details["dtype"]
    scale, zero_point = output_details.get("quantization", (0.0, 0))

    scores = np.squeeze(output_data)  # Remove batch dimension

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

**Softmax explained:**

```
Raw model outputs (logits):
[3.2, -1.5, 0.8, -0.3, 1.1]
  ↓
Softmax converts to probabilities:
[0.632, 0.006, 0.058, 0.019, 0.079]
  ↑
Sum = 1.0 (100%)

Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

### 7. Main Loop (Lines 98-162)

```python
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  "-m", required=True)
    parser.add_argument("--labels", "-l", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--top_k",  type=int, default=1)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    # Load model and labels
    labels = load_labels(args.labels)
    interpreter = make_interpreter(args.model)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps_avg = None
    while True:
        t0 = time.time()

        # Read frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and classify
        img_arr = preprocess_cv2(frame, input_shape)
        set_input_tensor(interpreter, img_arr)
        interpreter.invoke()
        probs = get_output_probs(interpreter)

        # Get top K predictions
        top_idx = np.argsort(probs)[-args.top_k:][::-1]
        predictions = [(labels[i], float(probs[i])) for i in top_idx]

        # Display predictions
        y0 = 30
        for i, (lbl, p) in enumerate(predictions):
            text = f"{lbl}: {p:.2f}"
            cv2.putText(frame, text, (10, y0 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Calculate and display FPS
        t1 = time.time()
        fps = 1.0 / (t1 - t0)
        fps_avg = fps if fps_avg is None else fps_avg * 0.9 + fps * 0.1
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Show frame
        cv2.imshow("TFLite Camera Inference", frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
```

## Command-Line Arguments

### Required Arguments

```bash
--model   (-m)  # Path to .tflite or .lite model file
--labels  (-l)  # Path to labels.txt
```

### Optional Arguments

```bash
--camera  # Camera index (default: 0)
          # 0 = built-in webcam
          # 1 = first USB camera

--top_k   # Number of predictions to show (default: 1)
          # 1 = only best prediction
          # 3 = top 3 predictions
          # 5 = top 5 predictions

--width   # Camera capture width (default: 640)
--height  # Camera capture height (default: 480)
```

## Usage Examples

### Example 1: Basic Usage
```bash
python camera_infer_tflite.py \
  --model models/model.lite \
  --labels labels.txt \
  --camera 0 \
  --top_k 3
```

**Output display:**
```
┌─────────────────────────────┐
│ ESP32: 0.89                 │  ← Top prediction
│ ESP8266: 0.07               │  ← 2nd best
│ Arduino: 0.03               │  ← 3rd best
│                             │
│  [Live camera feed]         │
│                             │
│                             │
│                   FPS: 45.2 │
└─────────────────────────────┘
```

### Example 2: High Resolution
```bash
python camera_infer_tflite.py \
  --model models/model.lite \
  --labels labels.txt \
  --width 1920 \
  --height 1080 \
  --top_k 1
```

### Example 3: External Camera
```bash
python camera_infer_tflite.py \
  --model models/model.lite \
  --labels labels.txt \
  --camera 1 \
  --top_k 5
```

## FPS Calculation with Smoothing

```python
# Exponential moving average
fps_avg = fps_avg * 0.9 + fps * 0.1
```

**Why smoothing:**
```
Without smoothing (raw FPS):
45, 47, 44, 48, 46, 45, ...  ← Jittery display

With smoothing (averaged):
45.0 → 45.2 → 45.1 → 45.4 → 45.5 → ...  ← Smooth display
```

## Performance Expectations

| Hardware | Expected FPS | Notes |
|----------|-------------|-------|
| MacBook Pro M1 | 80-120 | Excellent |
| Intel i7 Laptop | 50-80 | Very good |
| Intel i5 Laptop | 30-50 | Good |
| Raspberry Pi 4 | 15-25 | Acceptable |
| Raspberry Pi 3 | 5-10 | Slow but usable |

## Model Input Size Impact

```
Model Input Size  →  Speed       Accuracy
96x96            →  Very Fast   Lower
160x160          →  Fast        Good
224x224          →  Medium      Better
320x320          →  Slower      Very Good
512x512          →  Slow        Best
```

## Common Issues and Solutions

### Issue: Wrong colors in predictions
**Cause:** BGR vs RGB confusion
**Already handled:** Line 52 converts BGR→RGB

### Issue: Poor accuracy
**Solutions:**
1. Ensure good lighting
2. Hold object steady
3. Point camera directly at object
4. Check if object was in training data

### Issue: Low FPS
**Solutions:**
1. Reduce camera resolution
2. Use smaller model input size
3. Reduce top_k (less text to draw)
4. Close other applications

### Issue: Model not found
**Error:** `FileNotFoundError: Model not found`
**Solution:**
```bash
# Check model exists
ls -lh models/*.lite

# Use correct path
--model models/your-model-name.lite
```

## Comparison: Float32 vs Int8 Quantized

### Float32 Model (.h5 or float32.lite)
```
✅ Slightly better accuracy
❌ 4x larger file size
❌ 2-3x slower inference
❌ More memory usage
```

### Int8 Quantized Model (int8.lite)
```
✅ 4x smaller file size
✅ 2-3x faster inference
✅ Less memory usage
❌ Slightly less accurate (usually <1% difference)
```

**Recommendation:** Use quantized models for real-time applications

## Integration Example

```python
# Use in your own code
from camera_infer_tflite import make_interpreter, preprocess_cv2
from camera_infer_tflite import set_input_tensor, get_output_probs

# Load model once
interpreter = make_interpreter("model.lite")
labels = ["ESP32", "STM32", "Jetson"]

# Process single frame
frame = cv2.imread("test.jpg")
input_shape = interpreter.get_input_details()[0]['shape']
img_arr = preprocess_cv2(frame, input_shape)
set_input_tensor(interpreter, img_arr)
interpreter.invoke()
probs = get_output_probs(interpreter)

# Get prediction
best_idx = np.argmax(probs)
print(f"{labels[best_idx]}: {probs[best_idx]:.2%}")
```

## Related Files

- `camera_infer_h5.py`: Uses Keras .h5 models (unoptimized)
- `batch_infer_images_tflite.py`: Process saved images
- `batch_infer_h5.py`: Process saved images with .h5 models
- `labels.txt`: Class names for this model
