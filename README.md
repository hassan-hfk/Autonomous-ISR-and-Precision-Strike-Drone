# Drone Upload-Picture Tracker

SiamFC has been replaced with MobileCLIP-S0. MobileCLIP is Apple's edge-device
CLIP variant — small (~30 MB), fast on Jetson via TensorRT, and produces 512-dim
L2-normalized embeddings. Crucially, embedding the whole batch of candidate
detections in a single TensorRT call is way faster than SiamFC's per-patch loop.

## Build the MobileCLIP engine (one-time)

Export the model from the Apple `ml-mobileclip` package to ONNX first, then:

```bash
trtexec --onnx=mobileclip_s0_image_encoder.onnx \
        --saveEngine=mobileclip_s0_fp16.engine \
        --fp16 \
        --minShapes=images:1x3x256x256 \
        --optShapes=images:5x3x256x256 \
        --maxShapes=images:10x3x256x256
```

The dynamic shapes let us pass batches of any size from 1 to 10.

## Run

```bash
python3 upload_picture_tracker.py
```

## Files needed

- `yolov8n.engine` — YOLO TensorRT engine
- `mobileclip_s0_fp16.engine` — MobileCLIP image encoder
- `target.jpg`, `test_footage.mp4`

## What changed from v0.2

- SiamFC removed entirely (file deleted)
- New `mobileclip_embedder.py` with TensorRT wrapper
- Main script uses batched embedding (one call per frame, not one per detection)
- Fusion weights rebalanced: embed 0.55, hist 0.30, spatial 0.15
- Score threshold raised to 0.55 (MobileCLIP scores trend higher)
- Histogram comparison simplified to Bhattacharyya only (chi-square and
  correlation were redundant)

## TODO

- [ ] Sanity-check the FP16 engine output on real images
- [ ] Look at confidence calibration — are similar people getting similar scores?
- [ ] Click-track mode
