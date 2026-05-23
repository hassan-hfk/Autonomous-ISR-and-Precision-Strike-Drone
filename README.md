# Drone Upload-Picture Tracker

YOLO is now a TensorRT engine instead of an ONNX file. Massive speed boost
on Jetson — went from ~10 FPS effective to where SiamFC is now the
bottleneck. YOLO standalone runs at 60-80 FPS now.

## Building the engine (one-time)

```bash
yolo export model=yolov8n.pt format=engine device=0 half=True
# produces yolov8n.engine
```

Ultralytics auto-detects the .engine extension and uses TensorRT.

## How to run

Same as before:

```bash
python3 upload_picture_tracker.py
```

## Files needed

- `yolov8n.engine` — YOLO TensorRT engine (build with the export command above)
- `siamfc_alexnet.onnx` — SiamFC embedder
- `target.jpg` — reference image
- `test_footage.mp4` — video file

## What changed from v0.1

- YOLO_MODEL_PATH points to `.engine` instead of `.onnx`
- SKIP_FRAMES reduced from 5 to 2 (can afford more frequent rescoring)

## TODO

- [ ] Replace SiamFC — it's the bottleneck now, not YOLO
- [ ] Batch the similarity computation
- [ ] Click-track mode
