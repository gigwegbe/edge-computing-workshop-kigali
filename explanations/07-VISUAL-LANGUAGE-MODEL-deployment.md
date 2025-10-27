# Visual Language Model - deployment-script.py

**Location**: `visual-language-model/deployment-script.py`

## Purpose

Uses **LiquidAI's Vision-Language Model (VLM)** to analyze images and generate detailed textual descriptions. Unlike classification (which just identifies "ESP32") or detection (which finds boxes), VLMs can **understand and describe** what's in the image using natural language.

## What is a Visual Language Model (VLM)?

### Comparison with Other AI Models

```
IMAGE CLASSIFICATION              OBJECT DETECTION                  VISUAL LANGUAGE MODEL
Input: Image                      Input: Image                      Input: Image + Text Question
Output: "ESP32"                   Output: Boxes + Labels            Output: Natural language

┌──────────────┐                 ┌──────────────┐                 ┌──────────────┐
│              │                 │  ┌────┐      │                 │  [ESP32 img] │
│    ESP32     │ → "ESP32"       │  │ESP │      │                 │              │
│              │                 │  └────┘      │                 │ Question: Describe │
└──────────────┘                 └──────────────┘                 │ the board.        │
                                                                   │                   │
Single label                     Multiple boxes                   │ Answer: "A green  │
No description                   + labels                         │ ESP32 development │
                                                                   │ board with WiFi   │
                                                                   │ chip visible"     │
                                                                   └───────────────────┘
```

### How VLMs Work

```
Vision Encoder                    Language Model                   Combined Understanding
     ↓                                 ↓                                    ↓
┌──────────┐                    ┌──────────┐                       ┌──────────────┐
│  Image   │                    │  Text    │                       │ Multimodal   │
│ Features │ ──────────────────→│  Prompt  │ ─────────────────────→│ Analysis     │
│          │                    │          │                       │              │
└──────────┘                    └──────────┘                       └──────────────┘
 Visual understanding            Natural language                   Combines both
 (colors, shapes, text)          (understanding questions)          (generates answer)
```

## Key Concepts

### 1. Multimodal AI
- **Multi**: Multiple types of data
- **Modal**: Modalities (vision, language, audio)
- VLMs combine **vision** (images) and **language** (text)

### 2. Prompt Engineering
The quality of output depends heavily on **how you ask** the question:

```python
Bad prompt:
"Describe the image."
→ "There is a board in the image."

Good prompt:
"You are an expert electronics inspector. Describe the board,
including color, size, visible components, and any damage."
→ "A small green ESP32 development board with visible WiFi
antenna, USB port, and multiple GPIO pins. No damage detected."
```

### 3. JSON Structured Output
Instead of free-form text, request specific format:

```python
Prompt: "Return JSON with: description, color, size, confidence"

Output:
{
  "description": "ESP32 board with WiFi",
  "color": "green",
  "size": "small",
  "confidence": "high"
}
```

## Code Breakdown

### 1. Import Libraries (Lines 1-4)

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from PIL import Image
```

**Libraries explained:**

| Library | Purpose |
|---------|---------|
| `transformers` | Hugging Face library for AI models |
| `AutoProcessor` | Handles text and image preprocessing |
| `AutoModelForImageTextToText` | VLM model class |
| `PIL` | Python Image Library for loading images |

### 2. Load Model (Lines 7-15)

```python
# Model selection
model_id = "LiquidAI/LFM2-VL-450M"  # 450 million parameters

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",          # Automatically use GPU if available
    torch_dtype="bfloat16",     # Memory-efficient 16-bit precision
    trust_remote_code=True      # Allow custom model code
)

# Load processor (handles input preparation)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

**Model size options:**

| Model | Parameters | Memory | Speed | Quality |
|-------|-----------|--------|-------|---------|
| LFM2-VL-450M | 450M | ~2GB | Fast | Good |
| LFM2-VL-1.6B | 1.6B | ~6GB | Medium | Better |
| LFM2-VL-3B | 3B | ~12GB | Slow | Best |

**device_map="auto":**
- Automatically detects GPU (NVIDIA, Apple M1/M2)
- Falls back to CPU if no GPU
- Splits model across devices if needed

**torch_dtype="bfloat16":**
- Uses 16-bit floating point (instead of 32-bit)
- Reduces memory usage by 50%
- Minimal accuracy loss
- Not supported on all hardware (falls back to float32)

### 3. Load Test Image (Lines 18-22)

```python
image_path = "./detected/Esp32.683d77uh.ingestion-76f45fffcf-nn2ld_detections.jpg"
image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert("RGB")
```

**Why convert to RGB:**
```
Possible image modes:
- RGB: 3 channels (Red, Green, Blue) ✅ Model expects this
- RGBA: 4 channels (+ Alpha/transparency) ❌
- L: 1 channel (Grayscale) ❌
- CMYK: 4 channels (Cyan, Magenta, Yellow, Black) ❌

All converted to RGB for consistency
```

### 4. Create Conversation (Lines 26-45)

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": """You are an expert electronics board inspector.
Examine the image and verify whether the detected object(s) and their
reported confidence scores align with what you observe in the board.
Return the result strictly as JSON in the format below (no extra text, only JSON):
{
  "short description of damages": "<>",
  "confidence_level": "<>",
  "colors": "<Give one color>",
  "size": "<Give one size e.g small, medium, large>"
}
"""
            },
        ],
    },
]
```

**Conversation structure:**

```python
conversation = [
    {
        "role": "user",           # Who is speaking
        "content": [              # Message content
            {
                "type": "image",   # First part: image
                "image": <PIL Image>
            },
            {
                "type": "text",    # Second part: question
                "text": "<prompt>"
            }
        ]
    }
]
```

**Why this format:**
- Mimics chat conversation (like ChatGPT)
- Model trained on conversational data
- Can have multiple back-and-forth exchanges

### 5. Process Inputs (Lines 47-54)

```python
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,  # Add prompt to start model's response
    return_tensors="pt",         # Return PyTorch tensors
    return_dict=True,            # Return as dictionary
    tokenize=True,               # Convert text to tokens
).to(model.device)               # Move to GPU if available
```

**What processor does:**

```
1. Image Processing:
   Image → Resize → Normalize → Tensor
   (various sizes) → (model size) → (0-1 range) → (PyTorch)

2. Text Processing:
   Text → Tokenize → Add special tokens → Tensor
   "Describe board" → [101, 2847, 2604, 102] → <start>Describe board<end> → PyTorch

3. Combine:
   {
       "pixel_values": <image tensor>,
       "input_ids": <text tokens>,
       "attention_mask": <which tokens to attend to>
   }
```

**add_generation_prompt=True:**
```
Input prompt:  "User: Describe the board."
After adding:  "User: Describe the board.\nAssistant:"
               ↑ Model knows to complete after "Assistant:"
```

### 6. Generate Response (Lines 57-58)

```python
outputs = model.generate(**inputs, max_new_tokens=64 * 3)
decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
```

**max_new_tokens=192:**
```
Controls output length:
- Token ≈ 0.75 words
- 192 tokens ≈ 144 words
- Enough for detailed JSON response
```

**batch_decode:**
```
Model output:  [101, 284, 1567, ..., 102]  (token IDs)
               ↓
Decode:        "[CLS] A green ESP32... [SEP]"
               ↓
Remove special: "A green ESP32..."  ← Final text
```

### 7. Extract Response (Lines 60-64)

```python
# Extract only the assistant's reply
if "assistant" in decoded:
    response = decoded.split("assistant", 1)[1].strip()
else:
    response = decoded.strip()

print("\n--- Model Response ---")
print(response)
```

**Why split by "assistant":**
```
Full decoded output:
"User: Describe the board.\nAssistant: {\"description\": \"Green ESP32 board\", ...}"

After split:
"{\"description\": \"Green ESP32 board\", ...}"
↑ Only the assistant's answer
```

## Complete Workflow

```
1. Load VLM Model (~10-30 seconds)
   ├─ Download if not cached
   ├─ Load to GPU/CPU
   └─ Initialize processor
   ↓
2. Load Image
   ├─ Read from disk
   └─ Convert to RGB
   ↓
3. Create Conversation
   ├─ Define role: "user"
   ├─ Add image
   └─ Add text prompt
   ↓
4. Process Inputs
   ├─ Apply chat template
   ├─ Tokenize text
   ├─ Process image
   └─ Create attention masks
   ↓
5. Generate Response (~2-10 seconds)
   ├─ Run through model
   ├─ Generate tokens one by one
   └─ Stop at max_tokens or <end>
   ↓
6. Decode Output
   ├─ Convert tokens to text
   ├─ Remove special tokens
   └─ Extract assistant's response
   ↓
7. Display/Process Result
```

## Example Outputs

### Example 1: Good Detection
**Input Image:** Clear ESP32 board, green, well-lit

**Output:**
```json
{
  "short description of damages": "No visible damage, board appears intact",
  "confidence_level": "high",
  "colors": "green",
  "size": "small"
}
```

### Example 2: Damaged Board
**Input Image:** STM32 with burnt component

**Output:**
```json
{
  "short description of damages": "Visible burn marks on voltage regulator chip",
  "confidence_level": "high",
  "colors": "blue",
  "size": "medium"
}
```

### Example 3: Uncertain Detection
**Input Image:** Blurry Jetson board

**Output:**
```json
{
  "short description of damages": "Image quality insufficient for damage assessment",
  "confidence_level": "low",
  "colors": "green",
  "size": "large"
}
```

## Use Cases

### 1. Quality Control
```python
# Inspect boards for defects
prompt = """Check for: scratches, burnt components,
missing parts, solder issues. Return JSON."""
```

### 2. Inventory Management
```python
# Catalog boards with descriptions
prompt = """List all visible components,
board type, and condition. Return JSON."""
```

### 3. Automated Documentation
```python
# Generate documentation
prompt = """Create a technical description of this board
including visible interfaces and specifications."""
```

### 4. Educational Tool
```python
# Explain for beginners
prompt = """Explain what this board is and what it's used for
in simple terms a beginner can understand."""
```

## Running the Script

### Prerequisites
```bash
# Install required packages
pip install torch transformers accelerate Pillow

# For GPU support (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage
```bash
cd visual-language-model
python3 deployment-script.py
```

### Expected Output
```
Loading model...
[Download progress bars if first time]
Model loaded successfully!

Processing image...
Generating response...

--- Model Response ---
{
  "short description of damages": "No damage visible, board intact",
  "confidence_level": "high",
  "colors": "green",
  "size": "small"
}
```

## Performance

### First Run
```
Download model:  5-10 minutes (depending on internet)
Load model:      10-30 seconds
Generate:        3-10 seconds per image
Total:           ~15 minutes
```

### Subsequent Runs
```
Load model:      10-30 seconds (cached)
Generate:        3-10 seconds per image
Total:           ~15-40 seconds
```

### Hardware Requirements

| Hardware | Model Size | Performance |
|----------|-----------|-------------|
| CPU only | 450M | Slow (~30s/image) |
| NVIDIA GPU (8GB) | 450M | Fast (~3s/image) |
| NVIDIA GPU (16GB) | 1.6B | Fast (~5s/image) |
| NVIDIA GPU (24GB+) | 3B | Fast (~10s/image) |
| Apple M1/M2 | 450M | Medium (~10s/image) |

## Customization

### Change Model Size
```python
# Faster but less accurate
model_id = "LiquidAI/LFM2-VL-450M"

# Better but slower
model_id = "LiquidAI/LFM2-VL-1.6B"

# Best but requires 24GB GPU
model_id = "LiquidAI/LFM2-VL-3B"
```

### Change Prompt
```python
# Custom inspection prompt
text = """Inspect this electronic board.
Rate each category from 1-10:
- Physical condition
- Solder quality
- Component placement
Return as JSON."""
```

### Process Multiple Images
```python
import os
from pathlib import Path

# Get all detected images
detected_dir = Path("./detected")
images = list(detected_dir.glob("*.jpg"))

for img_path in images:
    print(f"\nAnalyzing: {img_path.name}")
    image = Image.open(img_path)
    # ... (rest of processing)
```

## Integration with Detection Pipeline

### Complete Workflow
```
1. Object Detection
   ├─ Input: Raw camera image
   ├─ Output: Detected boards with boxes
   └─ Save to: ./detected/
   ↓
2. VLM Analysis  (This script)
   ├─ Input: Detected images
   ├─ Output: Detailed descriptions
   └─ Save to: database/log
   ↓
3. Decision Making
   ├─ Parse JSON responses
   ├─ Flag damaged boards
   └─ Generate reports
```

### Example Integration
```python
# 1. Run object detection
from object_detection.detector import detect_objects
detected_path = detect_objects("input.jpg")

# 2. Analyze with VLM
from visual_language_model.deployment_script import analyze_board
description = analyze_board(detected_path)

# 3. Make decision
if "damage" in description["short description of damages"]:
    print("⚠️ Board needs inspection")
else:
    print("✅ Board passed quality check")
```

## Troubleshooting

### Issue: Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Use smaller model
model_id = "LiquidAI/LFM2-VL-450M"

# 2. Use CPU
device_map = "cpu"

# 3. Reduce max_tokens
max_new_tokens = 64  # Instead of 192
```

### Issue: Slow Generation
**Solutions:**
1. Use GPU instead of CPU
2. Use smaller model (450M)
3. Reduce max_new_tokens
4. Use quantized model (8-bit)

### Issue: Model Download Fails
**Solutions:**
```bash
# Set cache directory with more space
export HF_HOME=/path/to/large/disk/huggingface

# Or download manually
huggingface-cli download LiquidAI/LFM2-VL-450M
```

### Issue: Poor Quality Responses
**Solutions:**
1. Improve prompt engineering
2. Use larger model (1.6B or 3B)
3. Provide clearer instructions in prompt
4. Add examples in prompt (few-shot learning)

## Advanced Prompt Engineering

### Few-Shot Learning
```python
text = """You are an expert inspector.

Examples:
Image 1: {"damage": "none", "color": "green", "size": "small"}
Image 2: {"damage": "burn marks", "color": "blue", "size": "medium"}

Now analyze this image:"""
```

### Chain-of-Thought
```python
text = """Analyze step by step:
1. Identify the board type
2. Check for damage
3. Assess confidence
4. Return JSON

Format: {...}"""
```

## Related Files

- `object-detection/detector.py`: Creates images for VLM to analyze
- `object-detection/real-time-inference.py`: Real-time detection
- `detected/`: Folder with images to analyze
- `detected_others/`: Additional test images

## Future Enhancements

1. **Batch Processing**: Analyze multiple images in parallel
2. **Database Integration**: Store results in database
3. **Web Interface**: Upload images via web browser
4. **Real-time Integration**: Connect directly to camera feed
5. **Custom Fine-tuning**: Train on your specific boards
