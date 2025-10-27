# Object Detection - detector_nms.py

**Location**: `object-detection/detector_nms.py`

## Purpose

An **improved version** of `detector.py` that includes **Non-Maximum Suppression (NMS)** to eliminate duplicate bounding boxes on the same object. This produces cleaner, more professional detection results.

## What is Non-Maximum Suppression (NMS)?

### The Problem
Object detection models often predict **multiple overlapping boxes** for the same object:

```
Before NMS:
┌─────────┐
│┌────────┤
││ ESP32  │  ← 3 boxes on same board!
│└────────┘
└─────────┘

After NMS:
┌─────────┐
│  ESP32  │  ← Only the best box remains
└─────────┘
```

### The Solution
NMS keeps only the **highest confidence** box and removes others that overlap significantly:

1. Sort boxes by confidence score (highest first)
2. Keep the best box
3. Remove any boxes that overlap it too much (IoU > threshold)
4. Repeat for remaining boxes

## Key Differences from detector.py

### 1. Coordinate Conversion (Lines 85-90)
```python
# Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
boxes_xyxy = np.zeros_like(boxes)
boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
```

**Why needed:**
- Original boxes use center coordinates (cx, cy, width, height)
- NMS algorithm needs corner coordinates (x1, y1, x2, y2)
- Converts between these two formats

### 2. Non-Maximum Suppression (Lines 92-101)
```python
indices = tf.image.non_max_suppression(
    boxes=boxes_xyxy,
    scores=scores,
    max_output_size=50,        # Keep at most 50 boxes
    iou_threshold=0.45,        # Remove boxes with >45% overlap
    score_threshold=threshold  # Minimum confidence
).numpy()

# Keep only selected boxes
boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
```

**Parameters explained:**
- `max_output_size=50`: Maximum number of objects to detect
- `iou_threshold=0.45`: Overlap percentage that triggers removal
- `score_threshold`: Minimum confidence (typically 0.45 = 45%)

### 3. Model Size Difference
```python
# detector.py uses:
input_size = (96, 96)

# detector_nms.py uses:
input_size = (512, 512)
```

**Why the difference:**
- **512x512**: Higher resolution, better accuracy, slower inference
- **96x96**: Lower resolution, faster inference, slightly less accurate
- Choose based on your needs (accuracy vs speed)

## How NMS Works (Detailed)

### IoU (Intersection over Union) Calculation

```
Box A: ┌────────┐
       │ ┌──────┤───┐  ← Overlap
       │ │██████│   │
       └─┤██████│───┘
         └──────┘ Box B

IoU = Overlap Area / Total Area
```

### Algorithm Steps:

```python
# Simplified NMS pseudocode
sorted_boxes = sort_by_confidence(boxes)
keep = []

while sorted_boxes is not empty:
    best_box = sorted_boxes[0]
    keep.append(best_box)

    # Remove boxes that overlap too much
    for box in sorted_boxes[1:]:
        if iou(best_box, box) > threshold:
            remove(box)

return keep
```

### Example Values:

```
Original Detections:
- Box 1: ESP32, confidence=0.95, position=(100, 100, 150, 150)
- Box 2: ESP32, confidence=0.87, position=(105, 105, 155, 155)  ← 80% overlap!
- Box 3: STM32, confidence=0.92, position=(300, 300, 350, 350)

After NMS (threshold=0.45):
- Box 1: ESP32, confidence=0.95  ← Kept (highest confidence)
- Box 2: Removed (overlaps Box 1 by 80% > 45%)
- Box 3: STM32, confidence=0.92  ← Kept (different location)
```

## Complete Workflow

```
1. Load Model (512x512 YOLOv5)
   ↓
2. Load Image & Preprocess
   ↓
3. Run Inference → Get Raw Predictions
   ↓
4. Filter by Confidence (>45%)
   ↓
5. Convert Coordinates (center → corners)
   ↓
6. Apply NMS (remove duplicates)
   ↓
7. Draw Boxes on Original Image
   ↓
8. Save Annotated Image
```

## When to Use This vs detector.py

### Use `detector_nms.py` when:
- You need clean, professional-looking results
- Objects might be close together
- You're presenting results to others
- Accuracy is more important than speed

### Use `detector.py` when:
- You're doing quick tests
- Speed is critical
- Objects are far apart (less overlap)

## Performance Comparison

| Metric | detector.py | detector_nms.py |
|--------|-------------|-----------------|
| Input Size | 96x96 | 512x512 |
| Speed | ~100ms | ~300ms |
| Accuracy | Good | Better |
| Duplicate Boxes | Yes | No (removed) |
| Memory Usage | ~500MB | ~600MB |

## Model Information

- **Framework**: TensorFlow SavedModel
- **Architecture**: YOLOv5-512
- **Input**: 512x512 RGB images
- **Output**: Bounding boxes + class predictions
- **Trained on**: Edge Impulse platform
- **Dataset**: Images of ESP32, STM32, Jetson boards

## Code Structure

```
detector_nms.py
├── Import Libraries (Lines 1-12)
├── Load Model & Labels (Lines 14-25)
├── Define Colors (Lines 27-40)
├── detect_objects() function
│   ├── Load & Preprocess Image
│   ├── Run Inference
│   ├── Filter Predictions
│   ├── Apply NMS  ← New addition!
│   ├── Draw Boxes
│   └── Save Result
└── load_image() helper function
```

## Common Configuration

### Adjusting NMS Aggressiveness

```python
# More aggressive (removes more boxes):
iou_threshold=0.3  # Remove boxes with >30% overlap

# Less aggressive (keeps more boxes):
iou_threshold=0.6  # Remove only boxes with >60% overlap

# Default:
iou_threshold=0.45  # Balanced approach
```

### Changing Detection Confidence

```python
# More sensitive (more detections, more false positives):
threshold=0.3

# Less sensitive (fewer detections, fewer false positives):
threshold=0.6

# Default:
threshold=0.45  # Balanced
```

## Example Usage

```python
from detector_nms import detect_objects

# Detect with NMS
result_path = detect_objects(
    image_path="sample-directory/boards.jpg",
    output_dir="./clean_detections",
    threshold=0.5
)

print(f"Clean detections saved to: {result_path}")
```

## Troubleshooting

### Too Many Boxes Removed
**Problem**: NMS is too aggressive
**Solution**: Increase `iou_threshold` from 0.45 to 0.6

### Still Seeing Duplicates
**Problem**: NMS not aggressive enough
**Solution**: Decrease `iou_threshold` from 0.45 to 0.3

### Slow Performance
**Problem**: 512x512 input is large
**Solution**: Use `detector.py` with 96x96 instead

## Mathematical Details (Advanced)

### IoU Formula:
```
IoU = Area of Overlap / Area of Union

Where:
- Area of Overlap = intersection of two boxes
- Area of Union = total area covered by both boxes

IoU ∈ [0, 1]
- 0 = no overlap
- 1 = perfect overlap
```

### NMS Decision:
```
Keep Box B if: IoU(Box A, Box B) < threshold

Where:
- Box A = currently selected box
- Box B = candidate box
- threshold = 0.45 (45% overlap allowed)
```

## Related Files

- `detector.py`: Basic version without NMS
- `real-time-inference.py`: Camera feed without NMS
- `real-time-inference-tflite.py`: Camera feed WITH NMS (optimized)
