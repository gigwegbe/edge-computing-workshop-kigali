#!/usr/bin/env python3
"""
realtime_object_detection_tflite.py
Run a TensorFlow Lite (TFLite) object detection model on a webcam feed.
Applies Non-Maximum Suppression (NMS) to remove duplicate boxes and displays
bounding boxes and class labels in real-time.
"""

import cv2
import numpy as np
import tensorflow as tf
import time

# -----------------------------
# Load TFLite model
# -----------------------------
print("📦 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="./models/ei-edge-computing-workshop-2025-object-detection-yolov5-512-14.lite")
interpreter.allocate_tensors()
print("✅ Model loaded successfully!")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Load labels
# -----------------------------
labels = {}
with open("labels.txt", "r") as f:
    for i, line in enumerate(f.readlines()):
        labels[i] = line.strip()

# -----------------------------
# Colors for bounding boxes
# -----------------------------
colors = [
    (0, 255, 0), (0, 0, 255), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (0, 128, 128), (128, 128, 0)
]

# -----------------------------
# Non-Maximum Suppression (NMS)
# -----------------------------
def nms(boxes, scores, iou_threshold=0.45):
    """Perform Non-Maximum Suppression (NMS) using NumPy."""
    indices = []
    if len(boxes) == 0:
        return indices

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    while order.size > 0:
        i = order[0]
        indices.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return indices


def detect_objects_tflite(frame, threshold=0.45):
    """
    Run object detection using a TFLite model on a single frame.
    Returns annotated frame and detection metadata.
    """
    height, width, _ = frame.shape
    input_shape = input_details[0]['shape']
    input_size = (input_shape[2], input_shape[1])

    # Preprocess image
    resized = cv2.resize(frame, input_size)
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Assume output[0] is detections of shape (num_boxes, num_features)
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes = preds[:, 0:4]
    objectness = preds[:, 4]
    class_scores = preds[:, 5:]

    classes = np.argmax(class_scores, axis=-1)
    scores = objectness * np.max(class_scores, axis=-1)

    # Filter by confidence threshold
    mask = scores > threshold
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    # Convert boxes (cx, cy, w, h) → (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # NMS
    keep = nms(boxes_xyxy, scores, iou_threshold=0.45)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    # Draw detections
    for (x, y, w, h), score, cls in zip(boxes, scores, classes):
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        color = colors[cls % len(colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        class_name = labels.get(cls, str(cls))
        label = f"{class_name}: {score:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, boxes, scores, classes


def run_camera_inference(threshold=0.45, camera_index=0):
    """
    Run real-time object detection from webcam using a TFLite model.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    print("🎥 Starting camera stream... (press 'q' to quit)")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from camera")
            break

        # Detect objects
        annotated_frame, _, _, _ = detect_objects_tflite(frame, threshold)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("TFLite Real-Time Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_inference()
