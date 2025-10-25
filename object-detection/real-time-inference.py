#!/usr/bin/env python3
"""
realtime_object_detection.py
Run a TensorFlow SavedModel for object detection directly from a webcam feed.
Applies Non-Maximum Suppression (NMS) to remove duplicate boxes and overlays
bounding boxes and labels on live video.
"""

import tensorflow as tf
import numpy as np
import cv2
import time

# -----------------------------
# Load model once
# -----------------------------
print("📦 Loading TensorFlow model...")
detect_fn = tf.saved_model.load("./models/ei-edge-computing-workshop-2025-object-detection-yolov5-512/")
print("✅ Model loaded successfully!")

# -----------------------------
# Load labels once
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


def detect_objects_from_frame(frame, threshold=0.45):
    """
    Run object detection on a single frame.
    Returns annotated frame and detection metadata.
    """
    height, width, _ = frame.shape
    input_size = (512, 512)

    resized = cv2.resize(frame, input_size)
    input_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, axis=0) / 255.0

    detections = detect_fn(input_tensor)
    preds = detections[0].numpy()[0]  # (num_boxes, num_features)

    boxes = preds[:, 0:4]
    objectness = preds[:, 4]
    class_scores = preds[:, 5:]

    classes = np.argmax(class_scores, axis=-1)
    scores = objectness * np.max(class_scores, axis=-1)

    mask = scores > threshold
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    indices = tf.image.non_max_suppression(
        boxes=boxes_xyxy,
        scores=scores,
        max_output_size=50,
        iou_threshold=0.45,
        score_threshold=threshold
    ).numpy()

    boxes, scores, classes = boxes[indices], scores[indices], classes[indices]

    for (x, y, w, h), score, cls in zip(boxes, scores, classes):
        x1 = int((x - w/2) * width)
        y1 = int((y - h/2) * height)
        x2 = int((x + w/2) * width)
        y2 = int((y + h/2) * height)

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
    Open webcam feed and run real-time object detection.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open webcam")

    print("🎥 Starting camera stream... (press 'q' to quit)")

    prev_time = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from camera")
            break

        # Run detection
        annotated_frame, _, _, _ = detect_objects_from_frame(frame, threshold)

        # FPS calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_inference()
