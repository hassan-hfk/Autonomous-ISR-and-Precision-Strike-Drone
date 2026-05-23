# Drone Upload-Picture Tracker

First working version of the upload-picture tracker.

## What it does

Given:
- A reference image of a target (`target.jpg`)
- A video stream (`test_footage.mp4`)

It detects every object of the target class in each frame using YOLOv8n,
scores how similar each detection is to the reference image using a
combination of SiamFC features, HSV color histograms, spatial coherence,
and temporal smoothness, and locks onto the track ID that scores
consistently above threshold for several frames in a row.

## How to run

```bash
pip install ultralytics opencv-python onnxruntime-gpu numpy
python3 upload_picture_tracker.py
```

Press `q` to quit.

## Files needed in the same folder

- `yolov8n.onnx` — YOLO model (ONNX format)
- `siamfc_alexnet.onnx` — SiamFC embedder
- `target.jpg` — reference image of the target
- `test_footage.mp4` — video file to process

## Known limitations

- Effective rate is about 5-10 FPS — YOLO via ONNX runtime is slow on Jetson
- SiamFC is run patch-by-patch with no batching
- Three histogram comparison methods are averaged (probably overkill)
- Hard-coded paths and config (refactor later)

## TODO

- [ ] Export YOLO to TensorRT engine for speed
- [ ] Batch SiamFC inferences
- [ ] Move config to a separate file
- [ ] Click-track mode (in-flight click on arbitrary object)
