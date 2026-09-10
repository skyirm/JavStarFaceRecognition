import threading

import cv2
import numpy as np
import onnxruntime as ort

from config import ADAFACE_ONNX_PATH
from logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                options = ort.SessionOptions()
                options.enable_cpu_mem_arena = False
                options.log_severity_level = 3
                _session = ort.InferenceSession(
                    ADAFACE_ONNX_PATH, options, providers=["CPUExecutionProvider"]
                )
                logger.info("AdaFace loaded from %s", ADAFACE_ONNX_PATH)
    return _session


def preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 112x112 aligned crop -> RGB float32 NCHW in [-1, 1]."""
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 127.5 - 1.0
    return x.transpose(2, 0, 1)[None]


def get(aligned_bgr: np.ndarray) -> np.ndarray:
    """Aligned BGR face crop -> L2-normalized 512-d embedding."""
    session = _get_session()
    emb = session.run(None, {session.get_inputs()[0].name: preprocess(aligned_bgr)})[0][0]
    return emb / np.linalg.norm(emb)
