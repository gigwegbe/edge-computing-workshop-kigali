# Configuration - requirements.txt

**Location**: `requirements.txt`

## Purpose

Lists all Python packages (libraries) needed to run the workshop applications. This file is used by `pip` (Python's package installer) to automatically install all dependencies at once.

## What's Inside

```
numpy
tensorflow
Pillow
opencv-python
notebook
jupyter
accelerate
torch
transformers
huggingface_hub
```

## Package Explanations

### 1. numpy
**Purpose:** Fundamental package for numerical computing in Python

**What it does:**
- Array and matrix operations
- Mathematical functions
- Used by almost all AI/ML libraries

**Used in this project for:**
```python
# Image arrays
img_array = np.array([[[255, 0, 0], [0, 255, 0]]])  # 2x1 RGB image

# Mathematical operations
scores = np.exp(logits) / np.sum(np.exp(logits))  # Softmax

# Sorting and indexing
top_indices = np.argsort(probabilities)[-5:]  # Get top 5
```

**Size:** ~15MB

---

### 2. tensorflow
**Purpose:** Google's machine learning framework

**What it does:**
- Train machine learning models
- Run inference (predictions)
- Includes TensorFlow Lite for edge devices

**Used in this project for:**
```python
# Load models
model = tf.keras.models.load_model("model.h5")
detect_fn = tf.saved_model.load("saved_model/")

# TFLite inference
interpreter = tf.lite.Interpreter(model_path="model.lite")
interpreter.allocate_tensors()
interpreter.invoke()

# Operations
input_tensor = tf.convert_to_tensor(image)
nms_indices = tf.image.non_max_suppression(boxes, scores)
```

**Size:** ~500MB
**Note:** Includes CPU and GPU support

---

### 3. Pillow (PIL)
**Purpose:** Python Imaging Library for image manipulation

**What it does:**
- Load/save images
- Format conversion
- Basic image processing

**Used in this project for:**
```python
from PIL import Image

# Load image
img = Image.open("photo.jpg")

# Convert modes
img = img.convert("RGB")  # Ensure RGB format

# Resize
img = img.resize((320, 320), Image.BILINEAR)

# Convert to array
arr = np.asarray(img)
```

**Supported formats:**
- JPEG, PNG, BMP, GIF, TIFF, WebP, and more

**Size:** ~3MB

---

### 4. opencv-python
**Purpose:** Computer vision library (OpenCV)

**What it does:**
- Read/write images and video
- Real-time camera access
- Drawing shapes and text
- Image transformations

**Used in this project for:**
```python
import cv2

# Read image
img = cv2.imread("image.jpg")

# Camera access
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Draw on images
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, "ESP32", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# Display
cv2.imshow("Window", img)
cv2.waitKey(0)

# Save
cv2.imwrite("output.jpg", img)
```

**Size:** ~60MB

---

### 5. notebook
**Purpose:** Jupyter Notebook interface

**What it does:**
- Interactive Python environment
- Mix code, text, and visualizations
- Web-based coding interface

**Used for:**
- Experimenting with code
- Creating tutorials
- Data exploration

**Size:** ~10MB

---

### 6. jupyter
**Purpose:** Core Jupyter infrastructure

**What it does:**
- Jupyter server
- Notebook kernel management
- Extensions and plugins

**Used with `notebook` for:**
```bash
# Start Jupyter
jupyter notebook

# Opens browser with:
http://localhost:8888
```

**Size:** ~5MB

---

### 7. accelerate
**Purpose:** Hugging Face library for hardware acceleration

**What it does:**
- Automatically use GPU if available
- Distribute models across devices
- Mixed precision training

**Used in VLM script for:**
```python
# Automatically use best device (GPU/CPU)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto"  # ← accelerate handles this
)
```

**Size:** ~2MB

---

### 8. torch (PyTorch)
**Purpose:** Facebook's machine learning framework

**What it does:**
- Train machine learning models
- Run inference
- GPU acceleration
- Dynamic computation graphs

**Used in VLM script for:**
```python
import torch

# Model inference
outputs = model.generate(**inputs)

# Device management
model = model.to("cuda")  # Move to GPU

# Tensor operations
tensor = torch.tensor([[1, 2, 3]])
```

**Size:** ~750MB (includes CPU/GPU support)

**Alternatives:**
- `torch` (full version): CPU + GPU
- `torch-cpu`: CPU only (~200MB, faster install)

---

### 9. transformers
**Purpose:** Hugging Face library for pre-trained AI models

**What it does:**
- Access thousands of pre-trained models
- Text generation, translation
- Vision-language models
- Image classification

**Used in VLM script for:**
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

# Load VLM
processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-450M")
model = AutoModelForImageTextToText.from_pretrained("LiquidAI/LFM2-VL-450M")

# Process inputs
inputs = processor.apply_chat_template(conversation)

# Generate response
outputs = model.generate(**inputs)
decoded = processor.batch_decode(outputs)
```

**Size:** ~10MB (models downloaded separately)

---

### 10. huggingface_hub
**Purpose:** Interface to Hugging Face model hub

**What it does:**
- Download models
- Manage cache
- Authentication
- Model versioning

**Used for:**
```python
from huggingface_hub import hf_hub_download

# Download model file
model_path = hf_hub_download(
    repo_id="LiquidAI/LFM2-VL-450M",
    filename="model.safetensors"
)
```

**Size:** ~1MB

---

## Installation

### Basic Installation
```bash
# Install all packages
pip install -r requirements.txt
```

**Time:** 10-30 minutes (depending on internet speed)
**Space needed:** ~2GB

### With Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv edge-env

# Activate (Linux/Mac)
source edge-env/bin/activate

# Activate (Windows)
edge-env\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**Why use virtual environment:**
- Isolates project dependencies
- Avoids conflicts with other projects
- Easy to delete and recreate
- Portable

### Minimal Installation (Object Detection Only)
```bash
# Only essentials
pip install numpy tensorflow Pillow opencv-python
```

**Size:** ~600MB
**Time:** 5-10 minutes

### Minimal Installation (VLM Only)
```bash
# Only for VLM
pip install torch transformers accelerate Pillow huggingface_hub
```

**Size:** ~800MB
**Time:** 10-20 minutes

## Package Dependencies

### Dependency Tree
```
tensorflow
├── numpy
├── protobuf
├── six
└── ...

torch
├── numpy
└── ...

transformers
├── torch
├── numpy
├── huggingface_hub
├── tokenizers
└── ...

opencv-python
└── numpy
```

**Note:** `numpy` is a dependency of almost everything!

## Version Compatibility

### Checking Versions
```bash
# Check installed versions
pip list

# Specific package
pip show tensorflow
```

### Compatible Versions (Tested)
```
Python: 3.8 - 3.11
numpy: 1.19.0+
tensorflow: 2.8.0+
torch: 1.13.0+
transformers: 4.30.0+
opencv-python: 4.5.0+
```

### Common Conflicts

#### TensorFlow + NumPy
```bash
# Error: numpy version incompatible
# Solution:
pip install "numpy>=1.19.0,<2.0.0"
```

#### PyTorch + CUDA
```bash
# Issue: GPU not detected
# Solution: Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Platform-Specific Notes

### Linux
```bash
# Everything works out of the box
pip install -r requirements.txt
```

### macOS
```bash
# Use tensorflow-macos for M1/M2 (better performance)
pip install tensorflow-macos tensorflow-metal

# Then install others
pip install numpy Pillow opencv-python notebook jupyter accelerate torch transformers huggingface_hub
```

### Windows
```bash
# May need Visual C++ Redistributable
# Download from: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

# Then install
pip install -r requirements.txt
```

### Raspberry Pi
```bash
# Use lighter versions
pip install numpy
pip install tensorflow-lite-runtime  # Instead of full tensorflow
pip install Pillow opencv-python-headless  # Headless OpenCV (no GUI)

# Skip heavy packages
# Don't install: torch, transformers (too heavy for Pi)
```

## Storage Requirements

### Installed Packages
```
Total size: ~2GB

Breakdown:
├── tensorflow:        500MB
├── torch:             750MB
├── transformers:      10MB
├── opencv-python:     60MB
├── numpy:             15MB
├── Pillow:            3MB
├── accelerate:        2MB
├── huggingface_hub:   1MB
├── notebook:          10MB
└── jupyter:           5MB
```

### Downloaded Models (Separate)
```
Image Classification:  ~20MB
Object Detection:      ~50MB
VLM (LiquidAI):        ~2GB (downloaded on first run)

Total models:          ~2GB
```

### Total Project Space
```
Dependencies:    2GB
Models:          2GB
Code:            <1MB
Images (sample): ~50MB
──────────────────────
Total:           ~4GB
```

## Troubleshooting

### Issue: Slow pip install
**Solution:**
```bash
# Use faster mirror
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Issue: Package conflicts
**Solution:**
```bash
# Create fresh environment
python -m venv fresh-env
source fresh-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Out of space
**Solution:**
```bash
# Clean pip cache
pip cache purge

# Install one by one (monitor space)
pip install numpy
pip install tensorflow
# etc.
```

### Issue: Permission denied
**Solution:**
```bash
# Option 1: Use virtual environment (recommended)
python -m venv myenv
source myenv/bin/activate

# Option 2: User install (not recommended)
pip install --user -r requirements.txt
```

## Updating Packages

### Update all packages
```bash
pip install --upgrade -r requirements.txt
```

### Update specific package
```bash
pip install --upgrade tensorflow
```

### Check for updates
```bash
pip list --outdated
```

## Alternative: Conda Environment

### Using Conda instead of pip
```bash
# Create conda environment
conda create -n edge-workshop python=3.10

# Activate
conda activate edge-workshop

# Install packages
conda install numpy tensorflow pillow opencv jupyter
pip install accelerate torch transformers huggingface_hub
```

**Benefits:**
- Better dependency resolution
- Includes non-Python dependencies
- Easier environment management

## Related Files

- `.gitignore`: Lists `venv/` and `edge-env/` to exclude from git
- `README.md`: Setup instructions using this file
- `EI_Workshop_Windows.md`: Windows-specific setup guide
