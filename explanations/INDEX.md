# Complete Documentation Index

Welcome to the comprehensive documentation for the Edge Computing Workshop Kigali 2025! This index will guide you through all the explanations and help you understand every component of the project.

## Quick Start Guide

**New to the project?** Read these in order:
1. [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md) - Start here!
2. [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md) - Install dependencies
3. Choose your path:
   - Object Detection? → Files 01-04
   - Image Classification? → Files 05-06
   - Visual Language Models? → File 07

## Documentation Files

### 🎯 Overview and Setup

#### [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)
**READ THIS FIRST!**
- What is edge computing?
- Project structure overview
- Three main applications explained
- Key technologies and concepts
- How everything works together

**Topics covered:**
- Edge computing fundamentals
- Model optimization (Float32 vs TFLite vs Int8)
- Workflow examples
- Hardware requirements
- Getting started guide

---

#### [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
**Understanding Dependencies**
- Complete explanation of every Python package
- Why each package is needed
- Installation guide
- Platform-specific notes
- Troubleshooting

**Packages explained:**
- `numpy` - Array operations
- `tensorflow` - ML framework
- `opencv-python` - Computer vision
- `torch` - PyTorch for VLM
- `transformers` - Hugging Face models
- And 5 more...

---

### 🎯 Object Detection (Files 01-04)

Object detection finds and identifies specific objects in images, drawing bounding boxes around them.

#### [01-OBJECT-DETECTION-detector.md](01-OBJECT-DETECTION-detector.md)
**Basic Object Detection on Images**
- `object-detection/detector.py`
- Process saved images (batch)
- YOLOv5 model explained
- How bounding boxes work
- Input: Image file → Output: Annotated image

**Key concepts:**
- Model loading and inference
- Confidence thresholds
- Drawing bounding boxes
- Coordinate conversion

**When to use:**
- Analyzing saved photos
- Batch processing
- Offline analysis

---

#### [02-OBJECT-DETECTION-detector-nms.md](02-OBJECT-DETECTION-detector-nms.md)
**Improved Detection with NMS**
- `object-detection/detector_nms.py`
- Removes duplicate bounding boxes
- Higher resolution (512x512)
- Professional-quality results

**Key concepts:**
- Non-Maximum Suppression (NMS) explained
- Intersection over Union (IoU)
- When to use vs basic detector

**Diagram included:**
```
Before NMS: ┌─────┐    After NMS: ┌─────┐
            │┌────┤                │     │
            ││ESP │                │ ESP │
            │└────┘                │     │
            └─────┘                └─────┘
   3 overlapping boxes → 1 clean box
```

---

#### [03-OBJECT-DETECTION-real-time-inference.md](03-OBJECT-DETECTION-real-time-inference.md)
**Live Camera Object Detection**
- `object-detection/real-time-inference.py`
- Real-time webcam processing
- FPS calculation and display
- SavedModel format (TensorFlow)

**Key concepts:**
- Camera loop structure
- Frame-by-frame processing
- FPS optimization
- Continuous inference

**Performance:**
- 15-30 FPS typical
- ~50ms per frame
- Best for powerful computers

---

#### [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md)
**Optimized Real-time Detection**
- `object-detection/real-time-inference-tflite.py`
- 2-4x faster than SavedModel version
- TensorFlow Lite optimization
- Custom NMS implementation

**Key concepts:**
- TFLite vs TensorFlow comparison
- Model quantization explained
- Custom NumPy NMS
- Edge device optimization

**Performance:**
- 30-60 FPS typical
- ~25ms per frame
- Best for laptops and edge devices

**Advanced topics:**
- Operator fusion
- Graph optimization
- Memory efficiency

---

### 📸 Image Classification (Files 05-06)

Image classification categorizes entire images into predefined classes (e.g., "ESP32", "STM32").

#### [05-IMAGE-CLASSIFICATION-camera-infer-tflite.md](05-IMAGE-CLASSIFICATION-camera-infer-tflite.md)
**Real-time Image Classification**
- `image-classification/camera_infer_tflite.py`
- Live camera classification
- Top-K predictions
- TFLite optimized

**Key concepts:**
- Classification vs detection
- Softmax probability
- Top-K selection
- FPS smoothing

**Usage examples:**
```bash
# Show top 3 predictions
python camera_infer_tflite.py --model model.lite --labels labels.txt --top_k 3
```

**Outputs:**
```
ESP32: 0.89
STM32: 0.07
Arduino: 0.03
FPS: 45.2
```

---

#### [06-IMAGE-CLASSIFICATION-batch-infer-tflite.md](06-IMAGE-CLASSIFICATION-batch-infer-tflite.md)
**Batch Image Classification**
- `image-classification/batch_infer_images_tflite.py`
- Process saved images
- Single or directory input
- Inference time measurement

**Key concepts:**
- Batch vs real-time processing
- Quantization (Float32 vs Int8)
- Preprocessing pipeline
- Result interpretation

**Usage examples:**
```bash
# Single image
python batch_infer_images_tflite.py --model model.lite --labels labels.txt --image photo.jpg

# Directory
python batch_infer_images_tflite.py --model model.lite --labels labels.txt --images_dir ./photos/
```

**Output format:**
```
image1.jpg — inference 24.3 ms
  ESP32: 0.8852
  STM32: 0.0723
  Arduino: 0.0312
```

---

### 🤖 Visual Language Model (File 07)

VLMs understand images AND text, generating natural language descriptions.

#### [07-VISUAL-LANGUAGE-MODEL-deployment.md](07-VISUAL-LANGUAGE-MODEL-deployment.md)
**AI that Understands and Describes**
- `visual-language-model/deployment-script.py`
- LiquidAI LFM2-VL model
- Multimodal AI (vision + language)
- JSON structured output

**Key concepts:**
- What is a VLM?
- Multimodal AI explained
- Prompt engineering
- Chat template format

**Capabilities:**
```
Input:  Image + Question
Output: Detailed text description

Example:
Q: "Describe this board including color, size, and condition"
A: "Green ESP32 development board, small size, with visible
    WiFi chip and USB port. No damage detected. High confidence."
```

**Model sizes:**
- 450M parameters: Fast, good quality
- 1.6B parameters: Medium, better quality
- 3B parameters: Slow, best quality

**Advanced topics:**
- Prompt engineering best practices
- Few-shot learning
- JSON output formatting
- Integration with detection pipeline

---

## By Use Case

### 🔍 I want to detect objects in images
**Start here:**
1. Read [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)
2. Install packages: [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
3. For single images: [01-OBJECT-DETECTION-detector.md](01-OBJECT-DETECTION-detector.md)
4. For camera feed: [03-OBJECT-DETECTION-real-time-inference.md](03-OBJECT-DETECTION-real-time-inference.md)
5. For better performance: [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md)

### 📊 I want to classify images
**Start here:**
1. Read [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)
2. Install packages: [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
3. For saved images: [06-IMAGE-CLASSIFICATION-batch-infer-tflite.md](06-IMAGE-CLASSIFICATION-batch-infer-tflite.md)
4. For camera feed: [05-IMAGE-CLASSIFICATION-camera-infer-tflite.md](05-IMAGE-CLASSIFICATION-camera-infer-tflite.md)

### 💬 I want AI to describe what it sees
**Start here:**
1. Read [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)
2. Install packages (including torch): [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
3. Use VLM: [07-VISUAL-LANGUAGE-MODEL-deployment.md](07-VISUAL-LANGUAGE-MODEL-deployment.md)

### 🎓 I want to understand the concepts
**Read in order:**
1. [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md) - Big picture
2. [01-OBJECT-DETECTION-detector.md](01-OBJECT-DETECTION-detector.md) - Detection basics
3. [02-OBJECT-DETECTION-detector-nms.md](02-OBJECT-DETECTION-detector-nms.md) - NMS algorithm
4. [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md) - Optimization
5. [05-IMAGE-CLASSIFICATION-camera-infer-tflite.md](05-IMAGE-CLASSIFICATION-camera-infer-tflite.md) - Classification
6. [07-VISUAL-LANGUAGE-MODEL-deployment.md](07-VISUAL-LANGUAGE-MODEL-deployment.md) - Multimodal AI

---

## By Topic

### 📚 Core Concepts

**Edge Computing**
- File: [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md) → "What is Edge Computing?"
- What: Running AI on local devices instead of cloud
- Why: Faster, private, offline capability

**Model Optimization**
- File: [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md) → "Model Optimization"
- File: [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md) → "Optimization Techniques"
- Topics: Float32 → TFLite → Int8 quantization

**Object Detection vs Classification**
- File: [05-IMAGE-CLASSIFICATION-camera-infer-tflite.md](05-IMAGE-CLASSIFICATION-camera-infer-tflite.md) → "Image Classification vs Object Detection"
- Detection: WHERE objects are (boxes)
- Classification: WHAT the image is (labels)

---

### 🔧 Technical Details

**Non-Maximum Suppression (NMS)**
- File: [02-OBJECT-DETECTION-detector-nms.md](02-OBJECT-DETECTION-detector-nms.md)
- Algorithm explanation
- IoU calculation
- Code implementation

**TensorFlow Lite**
- File: [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md) → "Why TensorFlow Lite?"
- Conversion process
- Quantization
- Performance comparison

**Softmax and Probabilities**
- File: [05-IMAGE-CLASSIFICATION-camera-infer-tflite.md](05-IMAGE-CLASSIFICATION-camera-infer-tflite.md) → "Softmax explained"
- File: [06-IMAGE-CLASSIFICATION-batch-infer-tflite.md](06-IMAGE-CLASSIFICATION-batch-infer-tflite.md) → "Get Output Probabilities"
- Converting logits to probabilities

**Preprocessing Pipeline**
- All files include preprocessing sections
- BGR → RGB conversion
- Resizing
- Normalization
- Quantization

**Visual Language Models**
- File: [07-VISUAL-LANGUAGE-MODEL-deployment.md](07-VISUAL-LANGUAGE-MODEL-deployment.md)
- Multimodal AI
- Prompt engineering
- Token generation

---

### ⚙️ Practical Guides

**Installation and Setup**
- File: [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
- Package-by-package explanation
- Platform-specific instructions
- Troubleshooting

**Running Scripts**
- Every file includes:
  - Command-line arguments
  - Usage examples
  - Expected outputs
  - Troubleshooting

**Performance Optimization**
- File: [03-OBJECT-DETECTION-real-time-inference.md](03-OBJECT-DETECTION-real-time-inference.md) → "Performance Optimization Tips"
- File: [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md) → "Performance Comparison"
- FPS improvement strategies
- Memory management
- Hardware recommendations

---

## Quick Reference

### Command Cheat Sheet

```bash
# Object Detection (saved image)
python object-detection/detector.py

# Object Detection (camera, fast)
python object-detection/real-time-inference-tflite.py

# Image Classification (camera)
python image-classification/camera_infer_tflite.py \
  --model models/classifier.lite \
  --labels labels.txt \
  --top_k 3

# Image Classification (batch)
python image-classification/batch_infer_images_tflite.py \
  --model models/classifier.lite \
  --labels labels.txt \
  --images_dir ./photos/

# Visual Language Model
python visual-language-model/deployment-script.py
```

### File Locations

```
Project Root
├── object-detection/
│   ├── detector.py                    → File 01
│   ├── detector_nms.py                → File 02
│   ├── real-time-inference.py         → File 03
│   ├── real-time-inference-tflite.py  → File 04
│   ├── models/                        (trained models)
│   └── labels.txt                     (class names)
├── image-classification/
│   ├── camera_infer_tflite.py         → File 05
│   ├── batch_infer_images_tflite.py   → File 06
│   ├── batch_infer_h5.py              (unoptimized version)
│   ├── camera_infer_h5.py             (unoptimized version)
│   ├── models/                        (trained models)
│   └── labels.txt                     (class names)
├── visual-language-model/
│   └── deployment-script.py           → File 07
├── requirements.txt                   → File 08
└── explanations/                      (this folder!)
    ├── INDEX.md                       (you are here)
    ├── 00-PROJECT-OVERVIEW.md
    ├── 01-OBJECT-DETECTION-detector.md
    └── ... (all documentation files)
```

---

## Learning Path Recommendations

### 👶 Beginner (New to AI/ML)
1. **Week 1:** Understand concepts
   - Read [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)
   - Learn what edge computing is
   - Understand difference between detection and classification

2. **Week 2:** Run examples
   - Install dependencies: [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
   - Run object detection: [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md)
   - Try classification: [05-IMAGE-CLASSIFICATION-camera-infer-tflite.md](05-IMAGE-CLASSIFICATION-camera-infer-tflite.md)

3. **Week 3:** Understand code
   - Read through code sections in documentation
   - Follow preprocessing steps
   - Understand model inference

### 🎓 Intermediate (Some Python/ML experience)
1. **Day 1-2:** Overview and setup
   - Skim [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)
   - Install everything: [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)

2. **Day 3-4:** Object Detection deep dive
   - [01-OBJECT-DETECTION-detector.md](01-OBJECT-DETECTION-detector.md)
   - [02-OBJECT-DETECTION-detector-nms.md](02-OBJECT-DETECTION-detector-nms.md)
   - Understand NMS algorithm

3. **Day 5-6:** Optimization
   - [04-OBJECT-DETECTION-real-time-tflite.md](04-OBJECT-DETECTION-real-time-tflite.md)
   - Learn TFLite conversion
   - Compare performance

4. **Day 7:** Advanced topics
   - [07-VISUAL-LANGUAGE-MODEL-deployment.md](07-VISUAL-LANGUAGE-MODEL-deployment.md)
   - Explore VLM capabilities
   - Try prompt engineering

### 🚀 Advanced (ML Engineer/Researcher)
- All files have "Advanced" sections
- Focus on:
  - Model architecture details
  - Optimization techniques
  - Custom implementations
  - Integration patterns

---

## Additional Resources

### In the Project
- `README.md` - Quick start guide
- `EI_Workshop_Windows.md` - Windows setup
- `sample-directory/` - Test images
- `models/` - Pre-trained models

### External References
- Edge Impulse: https://edgeimpulse.com/
- TensorFlow Lite: https://www.tensorflow.org/lite
- PyTorch: https://pytorch.org/
- Hugging Face: https://huggingface.co/
- OpenCV: https://opencv.org/

---

## Glossary

**Edge Computing:** Running AI on local devices instead of cloud servers

**TensorFlow Lite:** Optimized version of TensorFlow for mobile/edge devices

**Quantization:** Reducing model precision (float32 → int8) for speed

**NMS (Non-Maximum Suppression):** Algorithm to remove duplicate bounding boxes

**IoU (Intersection over Union):** Measure of box overlap (0-1)

**FPS (Frames Per Second):** Processing speed metric

**Inference:** Running a trained model to make predictions

**Softmax:** Function that converts logits to probabilities

**VLM (Visual Language Model):** AI that understands both images and text

**Multimodal:** Processing multiple types of data (vision + language)

---

## Need Help?

### Troubleshooting Steps
1. Check if issue is mentioned in relevant documentation file
2. Look at "Common Issues" section in each file
3. Verify installation: [08-CONFIGURATION-requirements.md](08-CONFIGURATION-requirements.md)
4. Check hardware requirements: [00-PROJECT-OVERVIEW.md](00-PROJECT-OVERVIEW.md)

### Documentation Feedback
- Found an error? Note the filename and section
- Concept unclear? Which explanation needs improvement?
- Want more examples? For which script?

---

**Last Updated:** 2025-01-27

**Documentation Version:** 1.0

**Project:** Edge Computing Workshop Kigali 2025
