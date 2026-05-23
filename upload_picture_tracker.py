"""
Upload-picture tracker.

Given a reference image of a target, detect and track that specific object
in a video stream.

Pipeline:
  1. YOLOv8n (ONNX) detects all objects in current frame
  2. For each detection of target class, SiamFC scores similarity
     vs the uploaded target image
  3. Multi-modal fusion: SiamFC + HSV histogram + spatial + temporal
  4. Lock onto the YOLO track ID with consistent highest score
  5. If locked track lost for N frames, unlock and search again

Skip-frame: full similarity runs every 5 frames, in-between frames
reuse the last result.
"""
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import time
import os
import sys


# ============================================================================
# CONFIG
# ============================================================================

VIDEO_PATH = "test_footage.mp4"
TARGET_IMG = "target.jpg"

YOLO_MODEL_PATH = "yolov8n.onnx"
YOLO_CONF       = 0.30
YOLO_TARGET_CLS = 0   # 0 = person

SIAMFC_MODEL    = "siamfc_alexnet.onnx"
SIAMFC_TPL_SZ   = 127
SIAMFC_SRCH_SZ  = 255

# Multi-modal fusion weights
W_SIAMFC   = 0.15
W_HIST     = 0.60
W_SPATIAL  = 0.25
W_TEMPORAL = 0.10

# Lock parameters
MIN_CONSECUTIVE_MATCHES = 3
LOST_FRAMES_THRESHOLD   = 8
SCORE_THRESHOLD         = 0.45

# Skip-frame
SKIP_FRAMES = 5


# ============================================================================
# SIAMFC EMBEDDER (ONNX)
# ============================================================================

class SiamFCEmbedder:
    def __init__(self, model_path):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def embed(self, image_bgr, size):
        """Return feature vector for a crop."""
        if image_bgr is None or image_bgr.size == 0:
            return None
        img = cv2.resize(image_bgr, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # CHW with batch dim
        tensor = img.transpose(2, 0, 1)[None, ...]
        out = self.sess.run(None, {self.input_name: tensor})[0]
        # Global-average-pool the spatial dims
        feat = out.mean(axis=(2, 3)).flatten()
        feat /= (np.linalg.norm(feat) + 1e-8)
        return feat


# ============================================================================
# HISTOGRAM HELPERS
# ============================================================================

def hsv_histogram(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


def histogram_similarity(h1, h2):
    """Average of three OpenCV histogram comparison methods."""
    chi    = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
    corr   = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    bhatta = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    # normalize chi-square to 0..1, invert bhattacharyya (it's a distance)
    chi_n  = 1.0 / (1.0 + chi)
    bha_n  = 1.0 - bhatta
    return (chi_n + max(corr, 0) + bha_n) / 3.0


# ============================================================================
# CROP UTILS
# ============================================================================

def safe_crop(frame, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# ============================================================================
# MAIN TRACKER LOGIC
# ============================================================================

def main():
    # Load target
    target = cv2.imread(TARGET_IMG)
    if target is None:
        print(f"Failed to load {TARGET_IMG}")
        return 1

    # YOLO via ONNX
    yolo = YOLO(YOLO_MODEL_PATH)

    # SiamFC embedder
    embedder = SiamFCEmbedder(SIAMFC_MODEL)
    target_feat = embedder.embed(target, SIAMFC_TPL_SZ)
    target_hist = hsv_histogram(target)

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open {VIDEO_PATH}")
        return 1

    # State
    locked_tid = -1
    consecutive_matches = {}
    frames_lost = 0
    last_score_by_tid = {}
    last_center = None
    frame_idx = 0

    print("Starting tracker. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        # YOLO detect + track (uses ByteTrack internally via Ultralytics)
        results = yolo.track(frame, persist=True, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            cv2.imshow("track", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Filter to target class
        detections = []
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            if cls != YOLO_TARGET_CLS:
                continue
            tid = int(boxes.id[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].item())
            detections.append({'tid': tid, 'xyxy': xyxy, 'conf': conf})

        # If locked, find the lock and draw it
        if locked_tid != -1:
            found = next((d for d in detections if d['tid'] == locked_tid), None)
            if found:
                x1, y1, x2, y2 = [int(v) for v in found['xyxy']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"LOCKED tid={locked_tid}",
                            (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                frames_lost = 0
                last_center = bbox_center(found['xyxy'])
            else:
                frames_lost += 1
                if frames_lost >= LOST_FRAMES_THRESHOLD:
                    print(f"Lost track of tid={locked_tid}, unlocking")
                    locked_tid = -1
                    frames_lost = 0
                    consecutive_matches.clear()

        # Score candidates every SKIP_FRAMES frames OR when not locked
        if locked_tid == -1 and (frame_idx % SKIP_FRAMES == 0):
            for det in detections:
                crop = safe_crop(frame, det['xyxy'])
                if crop is None:
                    continue

                # SiamFC similarity
                feat = embedder.embed(crop, SIAMFC_TPL_SZ)
                if feat is None:
                    continue
                siam = float(np.dot(feat, target_feat))

                # Histogram similarity
                hist = hsv_histogram(crop)
                hist_sim = histogram_similarity(target_hist, hist)

                # Spatial coherence (closer to last known = better)
                if last_center is not None:
                    cx, cy = bbox_center(det['xyxy'])
                    dx = cx - last_center[0]
                    dy = cy - last_center[1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    spatial = max(0, 1.0 - dist / 200.0)
                else:
                    spatial = 0.5

                # Temporal smoothness — score similar to previous score
                prev = last_score_by_tid.get(det['tid'], None)
                if prev is not None:
                    temporal = max(0, 1.0 - abs(siam - prev) * 2)
                else:
                    temporal = 0.5

                combined = (W_SIAMFC * siam + W_HIST * hist_sim +
                            W_SPATIAL * spatial + W_TEMPORAL * temporal)
                last_score_by_tid[det['tid']] = combined

                # Draw all candidates
                x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
                color = (0, 0, 255)
                if combined >= SCORE_THRESHOLD:
                    color = (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{combined:.2f}",
                            (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Track consecutive matches
                if combined >= SCORE_THRESHOLD:
                    consecutive_matches[det['tid']] = \
                        consecutive_matches.get(det['tid'], 0) + 1
                else:
                    consecutive_matches[det['tid']] = 0

                # Lock if we've seen enough consecutive matches
                if consecutive_matches[det['tid']] >= MIN_CONSECUTIVE_MATCHES:
                    locked_tid = det['tid']
                    print(f"LOCKED onto tid={det['tid']} score={combined:.2f}")
                    break

        # FPS overlay
        dt = time.time() - t_start
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}  frame: {frame_idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("track", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
