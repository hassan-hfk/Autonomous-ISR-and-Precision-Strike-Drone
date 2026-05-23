"""
Upload-picture tracker.

MobileCLIP-S0 has replaced SiamFC for similarity scoring. The embeddings
are batched across all candidate detections per frame, which is a real
win over the per-patch SiamFC calls.

Pipeline:
  1. YOLOv8n (TensorRT engine) detects all objects in current frame
  2. For each detection of target class, collect crops
  3. MobileCLIP embeds the whole batch in one call
  4. Cosine similarity vs target embedding for each
  5. Fuse with HSV histogram and spatial coherence
  6. Lock onto YOLO track ID with consistent highest score
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import sys

from mobileclip_embedder import MobileCLIPEmbedder


# ============================================================================
# CONFIG
# ============================================================================

VIDEO_PATH = "test_footage.mp4"
TARGET_IMG = "target.jpg"

YOLO_MODEL_PATH = "yolov8n.engine"
YOLO_CONF       = 0.30
YOLO_TARGET_CLS = 0   # 0 = person

MOBILECLIP_ENGINE = "mobileclip_s0_fp16.engine"

# Fusion weights (rebalanced — MobileCLIP is stronger than SiamFC was)
W_EMBED   = 0.55
W_HIST    = 0.30
W_SPATIAL = 0.15

# Lock parameters
MIN_CONSECUTIVE_MATCHES = 3
LOST_FRAMES_THRESHOLD   = 8
SCORE_THRESHOLD         = 0.55

# Skip-frame
SKIP_FRAMES = 2


# ============================================================================
# HISTOGRAM HELPERS
# ============================================================================

def hsv_histogram(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return h


def histogram_similarity(h1, h2):
    """Bhattacharyya distance, inverted to a similarity score."""
    d = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return 1.0 - d


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
# MAIN
# ============================================================================

def main():
    target = cv2.imread(TARGET_IMG)
    if target is None:
        print(f"Failed to load {TARGET_IMG}")
        return 1

    yolo = YOLO(YOLO_MODEL_PATH)
    embedder = MobileCLIPEmbedder(MOBILECLIP_ENGINE)

    # Pre-compute target features
    target_feat = embedder.embed([target])[0]
    target_hist = hsv_histogram(target)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open {VIDEO_PATH}")
        return 1

    locked_tid = -1
    consecutive_matches = {}
    frames_lost = 0
    last_center = None
    frame_idx = 0

    print("Starting tracker. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()

        results = yolo.track(frame, persist=True, conf=YOLO_CONF, verbose=False)
        boxes = results[0].boxes

        if boxes is None or boxes.id is None:
            cv2.imshow("track", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        detections = []
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            if cls != YOLO_TARGET_CLS:
                continue
            tid = int(boxes.id[i].item())
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            detections.append({'tid': tid, 'xyxy': xyxy})

        # If locked, find and draw
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
                    print(f"Lost tid={locked_tid}, unlocking")
                    locked_tid = -1
                    frames_lost = 0
                    consecutive_matches.clear()

        # Score in batch when not locked
        if locked_tid == -1 and (frame_idx % SKIP_FRAMES == 0) and detections:
            crops = []
            valid_dets = []
            for det in detections:
                crop = safe_crop(frame, det['xyxy'])
                if crop is None:
                    continue
                crops.append(crop)
                valid_dets.append(det)

            if crops:
                # BATCH embed — this is the big win over SiamFC
                embeds = embedder.embed(crops)
                sims = embeds @ target_feat   # shape: (n,)

                for det, sim, crop in zip(valid_dets, sims, crops):
                    hist = hsv_histogram(crop)
                    hist_sim = histogram_similarity(target_hist, hist)

                    if last_center is not None:
                        cx, cy = bbox_center(det['xyxy'])
                        dx = cx - last_center[0]
                        dy = cy - last_center[1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        spatial = max(0, 1.0 - dist / 200.0)
                    else:
                        spatial = 0.5

                    combined = (W_EMBED * float(sim) +
                                W_HIST * hist_sim +
                                W_SPATIAL * spatial)

                    x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
                    color = (0, 0, 255)
                    if combined >= SCORE_THRESHOLD:
                        color = (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{combined:.2f} E={float(sim):.2f}",
                                (x1, max(y1 - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if combined >= SCORE_THRESHOLD:
                        consecutive_matches[det['tid']] = \
                            consecutive_matches.get(det['tid'], 0) + 1
                    else:
                        consecutive_matches[det['tid']] = 0

                    if consecutive_matches[det['tid']] >= MIN_CONSECUTIVE_MATCHES:
                        locked_tid = det['tid']
                        print(f"LOCKED tid={det['tid']} score={combined:.2f}")
                        break

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
