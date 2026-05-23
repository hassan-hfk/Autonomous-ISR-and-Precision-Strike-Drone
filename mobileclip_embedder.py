"""
MobileCLIP-S0 image encoder wrapped for use as a similarity scorer.

Built on TensorRT via pycuda. The engine was created with:

    trtexec --onnx=mobileclip_s0_image_encoder.onnx \\
            --saveEngine=mobileclip_s0_fp16.engine \\
            --fp16 \\
            --minShapes=images:1x3x256x256 \\
            --optShapes=images:5x3x256x256 \\
            --maxShapes=images:10x3x256x256

Inputs are BGR image crops, any size. They get resized to 256x256, converted
to RGB, normalized with CLIP's mean/std, and fed to the engine.

Outputs are L2-normalized 512-dim embeddings. Cosine similarity is just
np.dot(a, b).
"""
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # <-- creates its own CUDA context


# CLIP normalization stats
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class MobileCLIPEmbedder:
    def __init__(self, engine_path, max_batch=10):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.max_batch = max_batch

        # Pre-allocate buffers for max batch size
        self.input_shape = (max_batch, 3, 256, 256)
        self.output_shape = (max_batch, 512)
        self.h_input  = cuda.pagelocked_empty(int(np.prod(self.input_shape)),
                                              dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(int(np.prod(self.output_shape)),
                                              dtype=np.float32)
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream   = cuda.Stream()

    def _preprocess(self, crops):
        """List of BGR crops -> float32 NCHW batch."""
        batch = np.empty((len(crops), 3, 256, 256), dtype=np.float32)
        for i, c in enumerate(crops):
            if c is None or c.size == 0:
                batch[i] = 0
                continue
            img = cv2.resize(c, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = (img - MEAN) / STD
            batch[i] = img.transpose(2, 0, 1)
        return batch

    def embed(self, crops):
        if len(crops) == 0:
            return np.empty((0, 512), dtype=np.float32)
        if len(crops) > self.max_batch:
            raise ValueError(f"batch too large: {len(crops)} > {self.max_batch}")

        n = len(crops)
        batch = self._preprocess(crops)

        # Set dynamic shape and copy in
        self.context.set_input_shape("images", (n, 3, 256, 256))
        np.copyto(self.h_input[:n * 3 * 256 * 256], batch.flatten())

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_async_v2(bindings=bindings,
                                       stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        out = self.h_output[:n * 512].reshape(n, 512).copy()
        # L2 normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out / norms
