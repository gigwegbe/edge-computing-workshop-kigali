# Edge Computing Workshop Kigali 2025 - Project Overview

## What is This Project?

This is a comprehensive workshop project for learning about **Edge AI** and **Machine Learning** on edge devices. The project demonstrates how to deploy AI models (for image classification, object detection, and visual language understanding) on resource-constrained devices like laptops, Raspberry Pi, NVIDIA Jetson, and other edge computing platforms.

## Main Concepts

### 1. Edge Computing
Running AI models directly on local devices (laptops, embedded systems) instead of cloud servers. This provides:
- **Faster response times** (no network latency)
- **Privacy** (data stays on device)
- **Offline capability** (works without internet)
- **Lower costs** (no cloud computing fees)

### 2. Model Optimization
The project uses optimized models in different formats:
- **Float32 Models (.h5)**: Standard precision, larger size, slower
- **TensorFlow Lite (.lite)**: Optimized for mobile/edge devices
- **Int8 Quantized**: Compressed models, 4x smaller, faster inference

## Project Structure

```
edge-computing-workshop-kigali/
├── object-detection/          # Detect objects in images (ESP32, STM32, Jetson boards)
├── image-classification/      # Classify images into categories
├── visual-language-model/     # AI that understands and describes images
├── requirements.txt           # Python packages needed
├── README.md                  # Main documentation
└── explanations/             # This folder - detailed explanations
```

## Three Main Applications

### 1. Object Detection
**What it does**: Finds and identifies specific objects in images or video streams
**Use case**: Detecting electronic boards (ESP32, STM32, Jetson) in workshop images
**Models used**: YOLOv5 (You Only Look Once - a fast object detection algorithm)

### 2. Image Classification
**What it does**: Categorizes entire images into predefined classes
**Use case**: Identifying what type of board or component is in an image
**Models used**: CNN (Convolutional Neural Networks)

### 3. Visual Language Model (VLM)
**What it does**: Understands images and can answer questions or generate descriptions
**Use case**: Analyzing detected objects and providing detailed descriptions
**Models used**: LiquidAI LFM2-VL (multi-modal AI that processes both images and text)

## Workflow Example

Here's how the three applications work together:

```
1. CAMERA/IMAGE
   ↓
2. OBJECT DETECTION (finds boards in image)
   ↓
3. IMAGE CLASSIFICATION (identifies board type)
   ↓
4. VISUAL LANGUAGE MODEL (describes condition, color, size)
   ↓
5. RESULT: "ESP32 board detected with 95% confidence, green color, small size"
```

## Key Technologies Used

### Edge Impulse
A platform for building and deploying machine learning models on edge devices. It provides:
- Easy data collection
- Model training
- Model optimization
- Deployment tools

### TensorFlow / TensorFlow Lite
- **TensorFlow**: Google's machine learning framework
- **TensorFlow Lite**: Lightweight version for mobile and edge devices

### OpenCV
Computer vision library for:
- Reading images and video
- Processing frames
- Drawing bounding boxes
- Display results

### PyTorch & Transformers
- **PyTorch**: Facebook's machine learning framework
- **Transformers**: Library by Hugging Face for working with AI models like VLMs

## Hardware Supported

The models can run on:
- **Laptops/Desktops** (Windows, Mac, Linux)
- **NVIDIA Jetson** (Orin, Nano, Xavier)
- **Raspberry Pi**
- **Edge devices** with TensorFlow Lite support

## What You Downloaded (Models)

The `models/` folders contain:

1. **SavedModel format** (.pb files): TensorFlow's standard model format
2. **TensorFlow Lite models** (.lite files): Optimized for edge devices
3. **Model metadata** (.json files): Information about model performance

These models were trained using Edge Impulse on images of electronic boards.

## How to Use This Project

See the README.md for step-by-step instructions, but the general flow is:

1. **Set up environment**: Install Python and dependencies
2. **Choose an application**: Object detection, classification, or VLM
3. **Run inference**:
   - On saved images (batch processing)
   - On live camera feed (real-time)
4. **View results**: Annotated images or live video with predictions

## Next Steps

- Read individual file explanations in this folder
- Follow the README.md setup instructions
- Try running the different scripts
- Experiment with your own images

## Educational Value

This workshop teaches:
- How AI models work on edge devices
- The difference between different model formats
- Real-time vs batch inference
- Computer vision fundamentals
- How to deploy ML models practically
