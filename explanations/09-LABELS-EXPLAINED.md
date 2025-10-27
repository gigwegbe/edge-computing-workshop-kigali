# Understanding labels.txt Files

**Locations:**
- `object-detection/labels.txt`
- `image-classification/labels.txt`

## Purpose

Labels files map **class indices** (numbers) to **human-readable names**. Machine learning models work with numbers (0, 1, 2...), but we want to see meaningful names like "ESP32", "STM32", "Jetson".

## Format

Each line represents one class, with line number corresponding to class index:

```
ESP32
STM32
Jetson
Arduino
Raspberry Pi
Unknown
```

This means:
- Line 0 (first line) → Index 0 → "ESP32"
- Line 1 (second line) → Index 1 → "STM32"
- Line 2 (third line) → Index 2 → "Jetson"
- etc.

## How Models Use Them

### During Training
```
Training data:
- esp32_photo1.jpg  → labeled as "ESP32"   → encoded as 0
- stm32_photo1.jpg  → labeled as "STM32"   → encoded as 1
- jetson_photo1.jpg → labeled as "Jetson"  → encoded as 2

Model learns:
- Pattern A (WiFi chip visible, blue LED) → output 0
- Pattern B (STM32 text, blue board)      → output 1
- Pattern C (heatsink, large size)        → output 2
```

### During Inference (Prediction)

#### Object Detection Example
```python
# Model raw output
predictions = [
    [0.5, 0.5, 0.2, 0.3, 0.95, 0.05, 0.02, 0.88, 0.03, 0.01, 0.01],
    #                           \_____________________________/
    #                              class probabilities
]

# Find best class
class_scores = [0.05, 0.02, 0.88, 0.03, 0.01, 0.01]
best_class = 2  # Index of highest score (0.88)

# Load labels
labels = {0: "ESP32", 1: "STM32", 2: "Jetson", ...}

# Get human-readable name
class_name = labels[2]  # "Jetson"

# Display
print(f"Detected: {class_name} (88% confidence)")
# Output: Detected: Jetson (88% confidence)
```

#### Image Classification Example
```python
# Model output (softmax probabilities)
probabilities = [0.05, 0.23, 0.66, 0.04, 0.02]
#                ↑     ↑     ↑     ↑     ↑
#                0     1     2     3     4
#               ESP32 STM32 Jetson Ardu  RPi

# Get top prediction
best_idx = np.argmax(probabilities)  # 2 (highest value)

# Load labels
with open("labels.txt") as f:
    labels = [line.strip() for line in f]
# labels = ["ESP32", "STM32", "Jetson", "Arduino", "Raspberry Pi"]

# Display
print(f"{labels[best_idx]}: {probabilities[best_idx]:.2%}")
# Output: Jetson: 66%
```

## Reading labels.txt in Code

### Method 1: Dictionary (Object Detection)
```python
labels = {}
with open("labels.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        labels[i] = line.strip()

# Result:
# labels = {
#     0: "ESP32",
#     1: "STM32",
#     2: "Jetson",
#     ...
# }

# Usage:
class_name = labels[2]  # "Jetson"
```

### Method 2: List (Image Classification)
```python
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [l.strip() for l in f.readlines() if l.strip()]

# Result:
# labels = ["ESP32", "STM32", "Jetson", ...]

# Usage:
class_name = labels[2]  # "Jetson"
```

Both methods work the same way - index corresponds to model output!

## File Structure Details

### Example labels.txt
```
ESP32
STM32
Jetson Nano
```

**Important:**
- No extra spaces before/after
- No blank lines (or skip them in code)
- No numbers (just names)
- One name per line
- Order matters!

### Bad Examples

#### ❌ With line numbers
```
0 ESP32
1 STM32
2 Jetson
```
**Problem:** Model outputs 0, expects "ESP32", but file has "0 ESP32"

#### ❌ With extra blank lines
```
ESP32

STM32

Jetson
```
**Problem:** Blank line becomes index 1, "STM32" becomes index 2 (mismatch!)

#### ❌ Different order than training
```
Training order:        Inference order:
0: ESP32              0: STM32         ← Mismatch!
1: STM32              1: ESP32         ← Mismatch!
2: Jetson             2: Jetson        ✓
```
**Problem:** Model trained with ESP32=0, but labels.txt has STM32 first

## Creating Your Own labels.txt

### From Training Data
```python
# If you have training data organized like:
# dataset/
#   ├── esp32/
#   ├── stm32/
#   └── jetson/

import os
folders = sorted(os.listdir("dataset/"))
with open("labels.txt", "w") as f:
    for folder in folders:
        f.write(folder + "\n")
```

### From Model Metadata
```python
# Some models include labels in metadata
import json
with open("model_metadata.json") as f:
    metadata = json.load(f)
    labels = metadata["labels"]

# Save to file
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")
```

### Manual Creation
```bash
# Create file
echo "ESP32" > labels.txt
echo "STM32" >> labels.txt
echo "Jetson Nano" >> labels.txt
echo "Arduino" >> labels.txt
echo "Raspberry Pi" >> labels.txt
```

## Verifying labels.txt

### Check Number of Classes
```python
# Count lines
with open("labels.txt") as f:
    num_classes = len([l for l in f if l.strip()])
print(f"Number of classes: {num_classes}")

# Should match model's output size
# If model outputs 6 values, labels.txt should have 6 lines
```

### Check Order
```python
# Print with indices
with open("labels.txt") as f:
    for i, line in enumerate(f):
        if line.strip():
            print(f"Class {i}: {line.strip()}")

# Output:
# Class 0: ESP32
# Class 1: STM32
# Class 2: Jetson
# ...
```

### Compare with Training Data
```python
# Training labels
training_labels = ["ESP32", "STM32", "Jetson", "Arduino", "RaspberryPi"]

# Inference labels
with open("labels.txt") as f:
    inference_labels = [l.strip() for l in f if l.strip()]

# Check match
if training_labels == inference_labels:
    print("✓ Labels match!")
else:
    print("✗ Labels don't match!")
    print(f"Training: {training_labels}")
    print(f"Inference: {inference_labels}")
```

## Common Issues and Solutions

### Issue: Wrong predictions
```
Model detects:    "STM32" with 95% confidence
Actual object:    ESP32 board
```

**Possible cause:** labels.txt order doesn't match training
**Solution:**
```python
# Check model's training order
# Reorder labels.txt to match
# Or retrain model with correct order
```

### Issue: "Index out of bounds"
```python
IndexError: list index out of range
```

**Cause:** Model outputs more classes than labels.txt has
**Solution:**
```python
# Check model output size
output_size = model.output_shape[-1]  # e.g., 6

# Check labels.txt lines
num_labels = len(labels)  # e.g., 3

# labels.txt needs 6 lines to match model
```

### Issue: Garbled text
```
Model outputs: "ESP32ÿþ"
```

**Cause:** Encoding issues
**Solution:**
```python
# Always use UTF-8 encoding
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [l.strip() for l in f]
```

## Multi-language Support

### English + Local Language
```
ESP32 (Microcontroller)
STM32 (Development Board)
Jetson Nano (AI Computer)
```

### Different Languages
```python
# labels_en.txt
ESP32
STM32
Jetson

# labels_rw.txt  (Kinyarwanda)
Kibanza cya ESP32
Kibanza cya STM32
Mudasobwa Jetson

# Load based on preference
language = "en"  # or "rw"
with open(f"labels_{language}.txt") as f:
    labels = [l.strip() for l in f]
```

## Advanced: Labels with Metadata

### JSON Format (Alternative)
```json
{
  "labels": [
    {
      "id": 0,
      "name": "ESP32",
      "description": "ESP32 development board with WiFi",
      "color": "blue"
    },
    {
      "id": 1,
      "name": "STM32",
      "description": "STM32 ARM Cortex-M microcontroller",
      "color": "green"
    }
  ]
}
```

### Using JSON Labels
```python
import json

with open("labels.json") as f:
    data = json.load(f)

# Create lookup
labels = {item["id"]: item["name"] for item in data["labels"]}
descriptions = {item["id"]: item["description"] for item in data["labels"]}

# Usage
class_id = 0
print(f"Detected: {labels[class_id]}")
print(f"Description: {descriptions[class_id]}")
```

## Labels in Edge Impulse

This project uses **Edge Impulse** for model training. Edge Impulse automatically:

1. **Extracts labels** from training data folders
2. **Orders them** alphabetically or by upload order
3. **Exports** them in labels.txt format
4. **Embeds** them in model metadata

### Viewing Edge Impulse Labels
```bash
# Check what Edge Impulse used during training
cat object-detection/ei-*-json-file-*.json | grep -A 10 "labels"
```

## Integration Examples

### Display with Colors
```python
# Map labels to colors
label_colors = {
    "ESP32": (0, 255, 0),    # Green
    "STM32": (0, 0, 255),    # Red
    "Jetson": (255, 0, 0),   # Blue
}

# Load labels
with open("labels.txt") as f:
    labels = [l.strip() for l in f]

# Use in detection
for detection in detections:
    label = labels[detection.class_id]
    color = label_colors.get(label, (255, 255, 255))  # Default white
    cv2.rectangle(image, box, color, 2)
    cv2.putText(image, label, position, color=color)
```

### Logging
```python
import csv
from datetime import datetime

# Load labels
with open("labels.txt") as f:
    labels = [l.strip() for l in f]

# Log detections
with open("detections.csv", "a") as f:
    writer = csv.writer(f)
    for detection in detections:
        writer.writerow([
            datetime.now(),
            labels[detection.class_id],
            detection.confidence,
            detection.box
        ])
```

## Summary

**Key Points:**
1. labels.txt maps numbers to names
2. Line number = Class index
3. Order must match training
4. One label per line, no extras
5. Used by all inference scripts

**Common workflow:**
```
Training (Edge Impulse)
  ↓ Generates
labels.txt (automatic)
  ↓ Used by
Inference Scripts
  ↓ Display
Human-readable results
```

**Remember:**
- Model sees: `[0, 1, 2, 3, ...]`
- labels.txt translates: `["ESP32", "STM32", "Jetson", ...]`
- You see: `"Detected: ESP32 (95% confidence)"`

## Related Files

- `object-detection/labels.txt` - Object detection classes
- `image-classification/labels.txt` - Classification classes
- All Python scripts that use labels
- Edge Impulse model metadata (.json files)
