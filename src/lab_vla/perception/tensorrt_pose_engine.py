"""
TensorRT FP16 YOLO Pose Inference Engine for LabSOPGuard.

Provides:
- ONNX export from Ultralytics YOLO pose models
- TensorRT engine build with FP16 precision
- Optimized batch inference with TensorRT runtime
- Automatic fallback to PyTorch/Ultralytics when TensorRT unavailable

Target: ~200+ FPS per stream on RTX 3060+ with TensorRT FP16.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PoseDetection:
    """Single object detection with pose keypoints."""
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    keypoints: List[List[float]] = field(default_factory=list)  # [[x, y, conf], ...]
    keypoint_names: List[str] = field(default_factory=list)


@dataclass
class PoseInferenceResult:
    """Result from a single inference call."""
    frame_id: int
    timestamp_sec: float
    detections: List[PoseDetection]
    inference_time_ms: float
    backend: str  # "tensorrt", "onnxruntime", "ultralytics"
    image_size: Tuple[int, int] = (0, 0)  # (width, height)


@dataclass
class TensorRTConfig:
    """TensorRT engine configuration."""
    model_path: str  # .pt or .onnx source
    engine_path: str = ""  # Output .engine path
    fp16: bool = True
    int8: bool = False
    max_batch_size: int = 1
    workspace_size_mb: int = 512
    confidence_threshold: float = 0.25
    input_width: int = 640
    input_height: int = 640
    num_keypoints: int = 17  # COCO pose keypoints


def resolve_int8_calibration_cache(int8: bool, calibration_cache: str | Path | None) -> Optional[Path]:
    """Validate the optional INT8 calibration cache path."""
    if not int8:
        return None
    if not calibration_cache:
        raise RuntimeError("INT8 mode requires an existing calibration cache")
    cache_path = Path(calibration_cache).expanduser()
    if not cache_path.exists() or not cache_path.is_file():
        raise RuntimeError(f"INT8 calibration cache not found: {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model_path: str,
    output_path: str,
    input_width: int = 640,
    input_height: int = 640,
    opset: int = 12,
    simplify: bool = True,
) -> str:
    """Export YOLO pose model to ONNX format.

    Args:
        model_path: Path to .pt model file
        output_path: Output .onnx path
        input_width: Input width
        input_height: Input height
        opset: ONNX opset version
        simplify: Whether to simplify the ONNX model

    Returns:
        Path to exported ONNX file
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    onnx_path = Path(output_path)

    model.export(
        format="onnx",
        imgsz=[input_height, input_width],
        opset=opset,
        simplify=simplify,
        dynamic=False,
        half=False,  # Keep FP32 for ONNX, TensorRT will handle FP16
    )

    # Ultralytics exports to same name with .onnx extension
    auto_path = Path(model_path).with_suffix(".onnx")
    if auto_path.exists() and str(auto_path) != str(onnx_path):
        auto_path.rename(onnx_path)

    return str(onnx_path)


# ---------------------------------------------------------------------------
# TensorRT Engine Builder
# ---------------------------------------------------------------------------

def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    workspace_size_mb: int = 512,
) -> str:
    """Build TensorRT engine from ONNX model.

    Requires tensorrt Python package.
    Falls back gracefully if unavailable.

    Returns:
        Path to built engine file.
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise RuntimeError(
            "TensorRT Python package not installed. "
            "Install with: pip install tensorrt"
        )

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {errors}")

    # Builder config
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size_mb * 1024 * 1024

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        # TODO: Add INT8 calibration here for production

    # Set input shape
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    input_shape = (max_batch_size, 3, 640, 640)
    profile.set_shape(
        input_name,
        min=(1, 3, 640, 640),
        opt=(max_batch_size, 3, 640, 640),
        max=(max_batch_size, 3, 640, 640),
    )
    config.add_optimization_profile(profile)

    # Build engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT engine build failed")

    # Serialize and save
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    return engine_path


# ---------------------------------------------------------------------------
# TensorRT Inference Runner
# ---------------------------------------------------------------------------

class TensorRTPoseRunner:
    """TensorRT-based pose inference with FP16 optimization."""

    def __init__(self, engine_path: str, confidence_threshold: float = 0.25):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"TensorRT/PyCUDA not available: {e}")

        self.confidence_threshold = confidence_threshold
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_bindings):
            binding = self.engine.get_binding_shape(i)
            size = trt.volume(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({"host": host_mem, "device": device_mem})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem})

    def infer(self, frame_bgr: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> PoseInferenceResult:
        """Run TensorRT inference on a single frame."""
        import cv2
        import tensorrt as trt
        import pycuda.driver as cuda

        t0 = time.time()

        # Preprocess
        input_h, input_w = 640, 640
        img = cv2.resize(frame_bgr, (input_w, input_h))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim
        img = np.ascontiguousarray(img)

        # Copy to device
        np.copyto(self.inputs[0]["host"], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy back
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        t1 = time.time()
        inference_ms = (t1 - t0) * 1000

        # Parse output (YOLO pose output format)
        output = self.outputs[0]["host"].reshape(self.engine.get_binding_shape(1))
        detections = self._parse_output(output, frame_bgr.shape[1], frame_bgr.shape[0])

        return PoseInferenceResult(
            frame_id=frame_id,
            timestamp_sec=timestamp,
            detections=detections,
            inference_time_ms=inference_ms,
            backend="tensorrt",
            image_size=(frame_bgr.shape[1], frame_bgr.shape[0]),
        )

    def _parse_output(self, output: np.ndarray, img_w: int, img_h: int) -> List[PoseDetection]:
        """Parse YOLO pose output tensor to detections."""
        detections = []

        # Output shape: [1, 56, 8400] for YOLOv8-pose (4 bbox + 1 conf + 17*3 keypoints = 56)
        if output.ndim == 3:
            output = output[0]

        # Transpose to [8400, 56]
        if output.shape[0] < output.shape[1]:
            output = output.T

        for row in output:
            # Get confidence (5th value, index 4)
            conf = float(row[4])
            if conf < self.confidence_threshold:
                continue

            # Bbox: first 4 values
            x1, y1, x2, y2 = [float(v) for v in row[:4]]
            # Scale to original image
            x1 = x1 / 640 * img_w
            y1 = y1 / 640 * img_h
            x2 = x2 / 640 * img_w
            y2 = y2 / 640 * img_h

            # Keypoints: values 5 onwards, every 3 values = [x, y, conf]
            kps = []
            for i in range(5, min(len(row), 5 + 17 * 3), 3):
                kpx = float(row[i]) / 640 * img_w
                kpy = float(row[i + 1]) / 640 * img_h
                kpc = float(row[i + 2])
                kps.append([kpx, kpy, kpc])

            det = PoseDetection(
                label="person",
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                keypoints=kps,
                keypoint_names=[
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle",
                ],
            )
            detections.append(det)

        return detections


# ---------------------------------------------------------------------------
# Fallback: Ultralytics Inference
# ---------------------------------------------------------------------------

class UltralyticsPoseRunner:
    """Fallback pose inference using Ultralytics YOLO directly."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def infer(self, frame_bgr: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> PoseInferenceResult:
        t0 = time.time()
        results = self.model(frame_bgr, conf=self.confidence_threshold, verbose=False)
        t1 = time.time()

        detections = []
        for r in results:
            if r.keypoints is None:
                continue
            for i in range(len(r.boxes)):
                box = r.boxes[i]
                kp = r.keypoints[i]

                det = PoseDetection(
                    label=str(int(box.cls[0])),
                    confidence=float(box.conf[0]),
                    bbox=box.xyxy[0].tolist(),
                    keypoints=kp.xy[i].tolist() if kp.xy is not None else [],
                    keypoint_names=[
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle",
                    ],
                )
                detections.append(det)

        return PoseInferenceResult(
            frame_id=frame_id,
            timestamp_sec=timestamp,
            detections=detections,
            inference_time_ms=(t1 - t0) * 1000,
            backend="ultralytics",
            image_size=(frame_bgr.shape[1], frame_bgr.shape[0]),
        )


# ---------------------------------------------------------------------------
# Unified Pose Inference Engine
# ---------------------------------------------------------------------------

class PoseInferenceEngine:
    """Unified pose inference engine with automatic backend selection.

    Tries TensorRT -> ONNX Runtime -> Ultralytics in order.
    """

    def __init__(
        self,
        model_path: str,
        engine_path: str = "",
        fp16: bool = True,
        confidence_threshold: float = 0.25,
        force_backend: str = "",  # "tensorrt", "onnxruntime", "ultralytics", or "" for auto
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.runner = None
        self.backend = ""

        if force_backend == "ultralytics":
            self._init_ultralytics()
            return

        # Try TensorRT
        if force_backend in ("", "tensorrt"):
            try:
                self._init_tensorrt(engine_path, fp16)
                return
            except Exception:
                if force_backend == "tensorrt":
                    raise

        # Try ONNX Runtime
        if force_backend in ("", "onnxruntime"):
            try:
                self._init_onnxruntime(model_path)
                return
            except Exception:
                if force_backend == "onnxruntime":
                    raise

        # Fallback to Ultralytics
        self._init_ultralytics()

    def _init_tensorrt(self, engine_path: str, fp16: bool) -> None:
        """Initialize TensorRT backend."""
        ep = engine_path or str(Path(self.model_path).with_suffix(".engine"))

        if not Path(ep).exists():
            # Build engine from model
            onnx_path = str(Path(self.model_path).with_suffix(".onnx"))
            if not Path(onnx_path).exists():
                onnx_path = export_to_onnx(self.model_path, onnx_path)

            build_tensorrt_engine(onnx_path, ep, fp16=fp16)

        self.runner = TensorRTPoseRunner(ep, self.confidence_threshold)
        self.backend = "tensorrt"

    def _init_onnxruntime(self, model_path: str) -> None:
        """Initialize ONNX Runtime backend."""
        import onnxruntime as ort

        onnx_path = model_path
        if not onnx_path.endswith(".onnx"):
            onnx_path = str(Path(model_path).with_suffix(".onnx"))
            if not Path(onnx_path).exists():
                export_to_onnx(model_path, onnx_path)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(onnx_path, providers=providers)

        self.runner = _ONNXRunner(session, self.confidence_threshold)
        self.backend = "onnxruntime"

    def _init_ultralytics(self) -> None:
        """Initialize Ultralytics fallback."""
        self.runner = UltralyticsPoseRunner(self.model_path, self.confidence_threshold)
        self.backend = "ultralytics"

    def infer(self, frame_bgr: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> PoseInferenceResult:
        """Run inference on a frame."""
        return self.runner.infer(frame_bgr, frame_id, timestamp)

    def infer_batch(self, frames: List[np.ndarray], start_id: int = 0) -> List[PoseInferenceResult]:
        """Run inference on a batch of frames."""
        results = []
        for i, frame in enumerate(frames):
            result = self.infer(frame, frame_id=start_id + i)
            results.append(result)
        return results


class _ONNXRunner:
    """ONNX Runtime inference wrapper."""

    def __init__(self, session, confidence_threshold: float):
        self.session = session
        self.confidence_threshold = confidence_threshold
        self.input_name = session.get_inputs()[0].name

    def infer(self, frame_bgr: np.ndarray, frame_id: int = 0, timestamp: float = 0.0) -> PoseInferenceResult:
        import cv2

        t0 = time.time()
        img = cv2.resize(frame_bgr, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        output = self.session.run(None, {self.input_name: img})[0]
        t1 = time.time()

        # Parse output (same format as TensorRT)
        detections = []
        if output.ndim == 3:
            out = output[0]
        else:
            out = output

        if out.shape[0] < out.shape[1]:
            out = out.T

        h, w = frame_bgr.shape[:2]
        for row in out:
            conf = float(row[4])
            if conf < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = [float(v) / 640 * dim for v, dim in zip(row[:4], [w, h, w, h])]
            kps = []
            for i in range(5, min(len(row), 5 + 17 * 3), 3):
                kps.append([float(row[i]) / 640 * w, float(row[i+1]) / 640 * h, float(row[i+2])])
            detections.append(PoseDetection(
                label="person", confidence=conf, bbox=[x1, y1, x2, y2],
                keypoints=kps,
                keypoint_names=["nose","left_eye","right_eye","left_ear","right_ear",
                    "left_shoulder","right_shoulder","left_elbow","right_elbow",
                    "left_wrist","right_wrist","left_hip","right_hip",
                    "left_knee","right_knee","left_ankle","right_ankle"],
            ))

        return PoseInferenceResult(
            frame_id=frame_id, timestamp_sec=timestamp,
            detections=detections,
            inference_time_ms=(t1 - t0) * 1000,
            backend="onnxruntime",
            image_size=(w, h),
        )
