from __future__ import annotations

import av
import numpy as np

from .models import FrameAssembly, VideoFrame


class Decoder:
    def __init__(self) -> None:
        self._contexts: dict[str, av.codec.context.CodecContext] = {}

    def open(self) -> bool:
        return True

    def _get_context(self, stream_name: str) -> av.codec.context.CodecContext:
        key = stream_name or "rgb"
        ctx = self._contexts.get(key)
        if ctx is not None:
            return ctx
        ctx = av.CodecContext.create("h264", "r")
        self._contexts[key] = ctx
        return ctx

    @staticmethod
    def _decode_depth_from_bgr(image: np.ndarray) -> np.ndarray:
        low = image[:, :, 0].astype(np.uint16)
        high = image[:, :, 1].astype(np.uint16)
        return (high << 8) | low

    def decode(self, assembly: FrameAssembly) -> list[VideoFrame]:
        stream_name = assembly.stream_name or "rgb"
        ctx = self._get_context(stream_name)
        packet = av.Packet(assembly.payload)
        try:
            frames = ctx.decode(packet)
        except av.error.InvalidDataError:
            # Decoder may receive pre-IDR or partial recovery data during startup.
            return []
        out = []
        for index, frame in enumerate(frames):
            image = frame.to_ndarray(format="bgr24")
            if stream_name in ("depth_raw", "depth_aligned_to_rgb"):
                depth = self._decode_depth_from_bgr(image)
                pixel_fmt = "DEPTH16"
                data = depth
                width = depth.shape[1]
                height = depth.shape[0]
            else:
                pixel_fmt = "BGR8"
                data = image
                width = image.shape[1]
                height = image.shape[0]
            out.append(
                VideoFrame(
                    frame_id=assembly.frame_id if assembly.frame_id else index,
                    capture_ts_ms=assembly.capture_ts_ms,
                    width=width,
                    height=height,
                    pixel_fmt=pixel_fmt,
                    data=data,
                    sender_id=assembly.sender_id,
                    camera_id=assembly.camera_id,
                    channel_id=assembly.channel_id,
                    stream_name=stream_name,
                )
            )
        return out

    def close(self) -> None:
        self._contexts.clear()
