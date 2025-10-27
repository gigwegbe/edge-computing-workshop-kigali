# Object Detection - real-time-inference.py

**Location**: `object-detection/real-time-inference.py`

## Purpose

Performs **real-time object detection** on live video from your webcam. Detects objects (ESP32, STM32, Jetson boards) frame-by-frame and displays annotated video with bounding boxes and FPS counter.

## What Makes It "Real-Time"?

### Comparison with detector.py

| Feature | detector.py | real-time-inference.py |
|---------|-------------|------------------------|
| Input | Saved images | Live webcam |
| Processing | One image at a time | Continuous stream |
| Output | Saved image file | Live video display |
| Speed | Not critical | Very important |
| Use Case | Analysis | Demonstration |

## How It Works

### Main Loop Structure

```
Start Camera
   ↓
┌──────────────────┐
│ Read Frame       │ ← Continuous loop
│      ↓           │
│ Detect Objects   │
│      ↓           │
│ Draw Boxes       │
│      ↓           │
│ Calculate FPS    │
│      ↓           │
│ Display Frame    │
│      ↓           │
│ Check for 'q'    │ ← Press 'q' to quit
└──────────────────┘
```

## Key Components

### 1. Model Loading (Lines 17-18)
```python
print("📦 Loading TensorFlow model...")
detect_fn = tf.saved_model.load("./models/ei-edge-computing-workshop-2025-object-detection-yolov5-512/")
print("✅ Model loaded successfully!")
```

**Why load once:**
- Loading model takes several seconds
- Load once at startup, reuse for all frames
- Improves performance significantly

### 2. Camera Initialization (Lines 105-107)
```python
cap = cv2.VideoCapture(camera_index)  # Default: 0 (built-in webcam)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open webcam")
```

**Camera index values:**
- `0`: Built-in laptop webcam
- `1`: First external USB camera
- `2`: Second external camera
- etc.

### 3. Main Detection Function (Lines 39-97)

#### Frame Processing
```python
def detect_objects_from_frame(frame, threshold=0.45):
    height, width, _ = frame.shape
    input_size = (512, 512)

    # Resize frame to model's expected size
    resized = cv2.resize(frame, input_size)

    # Convert to tensor and normalize
    input_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0) / 255.0
```

**Key operations:**
1. Get original frame dimensions
2. Resize to 512x512 (model requirement)
3. Convert to TensorFlow tensor
4. Normalize: 0-255 → 0.0-1.0

#### Model Inference
```python
detections = detect_fn(input_tensor)
preds = detections[0].numpy()[0]

boxes = preds[:, 0:4]        # Location
objectness = preds[:, 4]     # Confidence
class_scores = preds[:, 5:]  # Class probabilities
```

#### Non-Maximum Suppression
```python
# Convert to corner coordinates
boxes_xyxy = np.zeros_like(boxes)
boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

# Apply NMS to remove duplicates
indices = tf.image.non_max_suppression(
    boxes=boxes_xyxy,
    scores=scores,
    max_output_size=50,
    iou_threshold=0.45,
    score_threshold=threshold
).numpy()
```

**Why NMS is crucial for real-time:**
- Video frames are similar between consecutive frames
- Without NMS, you'd see many overlapping boxes flickering
- NMS creates stable, clean bounding boxes

#### Drawing Annotations
```python
for (x, y, w, h), score, cls in zip(boxes, scores, classes):
    # Calculate pixel coordinates
    x1 = int((x - w/2) * width)
    y1 = int((y - h/2) * height)
    x2 = int((x + w/2) * width)
    y2 = int((y + h/2) * height)

    # Draw colored rectangle
    color = colors[cls % len(colors)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Add label
    class_name = labels.get(cls, str(cls))
    label = f"{class_name}: {score:.2f}"
    cv2.putText(frame, label, (x1, y1), ...)
```

### 4. Main Loop (Lines 114-137)

```python
prev_time = 0

while True:
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        print("⚠️ Failed to read frame from camera")
        break

    # Detect objects
    annotated_frame, _, _, _ = detect_objects_from_frame(frame, threshold)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display FPS on frame
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame in window
    cv2.imshow("Real-Time Object Detection", annotated_frame)

    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break
```

**Loop breakdown:**
1. **Read frame**: Get image from camera
2. **Detect**: Run object detection
3. **Calculate FPS**: Measure performance
4. **Display**: Show annotated frame
5. **Check input**: Look for 'q' key press

### 5. Cleanup (Lines 139-140)
```python
cap.release()            # Release camera
cv2.destroyAllWindows()  # Close display windows
```

**Why important:**
- Frees camera for other applications
- Closes OpenCV windows properly
- Prevents resource leaks

## FPS (Frames Per Second) Calculation

### Formula
```python
FPS = 1 / (current_time - previous_time)
```

### Example Calculation
```
Frame 1: time = 0.000s
Frame 2: time = 0.033s  →  FPS = 1 / 0.033 = 30 FPS
Frame 3: time = 0.066s  →  FPS = 1 / 0.033 = 30 FPS
```

### What Good FPS Looks Like

| FPS Range | Quality | Use Case |
|-----------|---------|----------|
| 60+ FPS | Excellent | Smooth real-time |
| 30-60 FPS | Good | Acceptable real-time |
| 15-30 FPS | Fair | Usable but choppy |
| <15 FPS | Poor | Too slow |

### Factors Affecting FPS

1. **Hardware**
   - CPU speed
   - Available RAM
   - GPU (if TensorFlow uses it)

2. **Model**
   - Input size (512x512 is large)
   - Model complexity
   - Quantization (int8 vs float32)

3. **Resolution**
   - Camera resolution
   - Display resolution

## Running the Script

### Basic Usage
```bash
cd object-detection
python3 real-time-inference.py
```

### What You'll See

```
📦 Loading TensorFlow model...
✅ Model loaded successfully!
🎥 Starting camera stream... (press 'q' to quit)

[Window opens showing live camera feed with:]
┌─────────────────────────────────┐
│ FPS: 28.3                       │  ← Performance indicator
│                                 │
│     ┌──────────┐                │
│     │ ESP32    │                │  ← Detected object
│     │ 0.89     │                │  ← Confidence
│     └──────────┘                │
│                                 │
└─────────────────────────────────┘
```

### To Exit
- Press `q` on keyboard
- Or press `Ctrl+C` in terminal

## Performance Optimization Tips

### If FPS is too low (<15):

1. **Use the TFLite version instead**
   ```bash
   python3 real-time-inference-tflite.py
   ```
   TFLite is optimized for edge devices (faster)

2. **Reduce camera resolution**
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

3. **Increase confidence threshold**
   ```python
   detect_objects_from_frame(frame, threshold=0.6)  # Process fewer boxes
   ```

4. **Close other applications**
   Free up CPU/RAM

## Common Issues and Solutions

### Camera Not Opening
**Error:** `❌ Cannot open webcam`

**Solutions:**
1. Check if camera is already in use (Zoom, Skype, etc.)
2. Try different camera index:
   ```python
   run_camera_inference(camera_index=1)
   ```
3. Check camera permissions (especially on Mac)

### Poor Performance
**Symptom:** FPS < 10

**Solutions:**
1. Use TFLite version (faster)
2. Use smaller input size model
3. Upgrade hardware
4. Use GPU acceleration (if available)

### No Detections
**Symptom:** Video shows but no boxes

**Solutions:**
1. Point camera at trained objects (ESP32, STM32, Jetson)
2. Ensure good lighting
3. Lower confidence threshold
4. Check model path is correct

### Flickering Bounding Boxes
**Symptom:** Boxes appear/disappear rapidly

**Solutions:**
1. This is expected with borderline confidence scores
2. Increase confidence threshold (fewer but stable boxes)
3. NMS is already applied, so some flicker is normal

## Code Structure

```
real-time-inference.py
├── Imports (Lines 1-12)
├── Model Loading (Lines 15-27)
├── detect_objects_from_frame()
│   ├── Preprocess frame
│   ├── Run inference
│   ├── Apply NMS
│   ├── Draw annotations
│   └── Return annotated frame
├── run_camera_inference()
│   ├── Initialize camera
│   ├── Main loop
│   │   ├── Read frame
│   │   ├── Detect objects
│   │   ├── Calculate FPS
│   │   ├── Display frame
│   │   └── Check for quit
│   └── Cleanup
└── Main execution
```

## Comparison with TFLite Version

| Feature | real-time-inference.py | real-time-inference-tflite.py |
|---------|------------------------|-------------------------------|
| Model Format | SavedModel (.pb) | TFLite (.lite) |
| Speed | Slower (~15-30 FPS) | Faster (~30-60 FPS) |
| Memory | More (~600MB) | Less (~200MB) |
| Accuracy | Slightly better | Slightly less |
| Best For | Desktop with GPU | Laptops, edge devices |

## Real-World Applications

1. **Quality Control**
   - Inspect boards on assembly line
   - Real-time defect detection

2. **Inventory Management**
   - Count boards automatically
   - Identify board types

3. **Education**
   - Demonstrate AI concepts
   - Workshop presentations

4. **Prototyping**
   - Test detection models
   - Collect training data

## Related Files

- `real-time-inference-tflite.py`: Optimized version (faster)
- `detector.py`: Single image version
- `detector_nms.py`: Batch processing with NMS
- `models/`: Contains the YOLOv5 model
- `labels.txt`: Object class names
