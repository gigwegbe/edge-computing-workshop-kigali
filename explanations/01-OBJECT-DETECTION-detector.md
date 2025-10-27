# Object Detection - detector.py

**Location**: `object-detection/detector.py`

## Purpose

This script performs **object detection** on individual images saved on your computer. It identifies and locates objects (like ESP32, STM32, Jetson boards) in images and saves the results with bounding boxes drawn around detected objects.

## What is Object Detection?

Unlike image classification (which labels the entire image), object detection:
1. **Finds** where objects are in the image (bounding boxes)
2. **Identifies** what each object is (class labels)
3. **Calculates** confidence scores for each detection

## How It Works

### Step-by-Step Process

```
1. Load Image → 2. Resize → 3. Normalize → 4. Run Model → 5. Filter Results → 6. Draw Boxes → 7. Save
```

### Detailed Breakdown

#### 1. Model Loading (Lines 10)
```python
detect_fn = tf.saved_model.load("./models/ei-edge-computing-workshop-2025-object-detection-yolov5-512/")
```
- Loads the YOLOv5 model trained on Edge Impulse
- This is a **SavedModel format** (TensorFlow's standard format)
- Model is loaded once at startup for efficiency

#### 2. Labels Loading (Lines 15-18)
```python
labels = {}
with open("labels.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        labels[i] = line.strip()
```
- Reads class names from `labels.txt`
- Maps numbers (0, 1, 2) to names ("ESP32", "STM32", "Jetson")

#### 3. Image Preprocessing (Lines 48-58)
```python
image = cv2.imread(image_path)
height, width, _ = image.shape
input_size = (96, 96)
resized_image = cv2.resize(image, input_size)
input_tensor = tf.convert_to_tensor(resized_image, dtype=tf.float32)
input_tensor = tf.expand_dims(input_tensor, axis=0) / 255.0
```

**What happens here:**
- Load image using OpenCV
- Store original dimensions (for drawing boxes later)
- Resize to **96x96** pixels (what the model expects)
- Convert to TensorFlow tensor
- Normalize pixel values from 0-255 to 0.0-1.0
- Add batch dimension (shape becomes [1, 96, 96, 3])

#### 4. Model Inference (Lines 60-67)
```python
detections = detect_fn(input_tensor)
preds = detections[0].numpy()[0]  # Shape: (num_boxes, 11)

boxes = preds[:, 0:4]        # x, y, width, height
objectness = preds[:, 4]     # confidence that something is there
class_scores = preds[:, 5:]  # probabilities for each class
```

**Model output explained:**
- Returns predictions for multiple potential objects
- Each prediction has 11 values:
  - 4 values for box location (x, y, width, height)
  - 1 value for objectness (is there an object here?)
  - 6 values for class probabilities (one per class)

#### 5. Score Calculation (Lines 69-71)
```python
classes = np.argmax(class_scores, axis=-1)
scores = objectness * np.max(class_scores, axis=-1)
```
- Find which class has highest probability
- Multiply objectness by class score for final confidence

#### 6. Filtering (Lines 73-75)
```python
mask = scores > threshold  # Default 0.45 (45%)
boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
```
- Only keep detections with confidence > 45%
- Removes low-confidence false positives

#### 7. Drawing Boxes (Lines 77-93)
```python
for (x, y, w, h), score, cls in zip(boxes, scores, classes):
    # Convert normalized coordinates to pixels
    x1 = int((x - w/2) * width)
    y1 = int((y - h/2) * height)
    x2 = int((x + w/2) * width)
    y2 = int((y + h/2) * height)

    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Add label with confidence
    label = f"{class_name}: {score:.2f}"
    cv2.putText(image, label, (x1, y1), ...)
```

**Key points:**
- Converts center coordinates (x, y, w, h) to corner coordinates (x1, y1, x2, y2)
- Scales from model's 96x96 space back to original image size
- Draws colored rectangles (each class gets a different color)
- Adds text label showing class name and confidence score

#### 8. Saving Output (Lines 95-101)
```python
output_path = os.path.join(output_dir, f"{filename}_detections.jpg")
cv2.imwrite(output_path, image)
```
- Creates `detected/` directory if it doesn't exist
- Saves annotated image with "_detections" suffix

## Key Functions

### `detect_objects(image_path, output_dir, threshold)`
**Parameters:**
- `image_path`: Path to image to analyze
- `output_dir`: Where to save result (default: "./detected")
- `threshold`: Minimum confidence (default: 0.45 = 45%)

**Returns:**
- Path to the saved annotated image

### `load_image(image_path)`
**Purpose:** Load and convert image to RGB format
**Returns:** PIL Image object

## Colors Used

Each object class gets a unique color for visualization:
- Green: ESP32
- Red: STM32
- Blue: Jetson
- Cyan, Magenta, Yellow: Other classes (if added later)

## Model Specifications

- **Architecture**: YOLOv5 (You Only Look Once version 5)
- **Input size**: 96x96 pixels RGB
- **Output**: Multiple bounding boxes with class predictions
- **Classes detected**: 6 (likely ESP32, STM32, Jetson, and variations)

## When to Use This Script

- **Batch processing**: Analyze multiple saved images
- **Offline analysis**: No camera needed
- **Quality inspection**: Carefully examine detection results
- **Testing**: Verify model accuracy before real-time deployment

## Example Usage

```bash
# In Python code
from detector import detect_objects

# Detect objects in a single image
output = detect_objects(
    image_path="sample-directory/board1.jpg",
    output_dir="./results",
    threshold=0.5
)
print(f"Result saved to: {output}")
```

## Common Issues and Solutions

### Issue: Low confidence scores
**Solution**: Lower threshold (e.g., 0.3) or retrain model with more data

### Issue: Multiple boxes on same object
**Solution**: Use `detector_nms.py` instead (has Non-Maximum Suppression)

### Issue: No detections
**Solution**: Check if image contains trained objects, verify model path

## Performance

- **Speed**: ~100-200ms per image (depends on hardware)
- **Accuracy**: Depends on training data quality
- **Memory**: ~500MB for model

## Related Files

- `detector_nms.py`: Same functionality + removes duplicate boxes
- `real-time-inference.py`: Uses this model on camera feed
- `labels.txt`: Class names used by this script
