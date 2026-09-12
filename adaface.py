import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from config import ADAFACE_ONNX_PATH
from logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_session: ort.InferenceSession | None = None

_MODEL_PATH = Path(ADAFACE_ONNX_PATH)
if not _MODEL_PATH.is_absolute():
    _MODEL_PATH = Path(__file__).resolve().parent / _MODEL_PATH


def _download_model() -> None:
    """Download the ONNX model when missing. Source URL: env ADAFACE_ONNX_URL."""
    url = os.environ.get("ADAFACE_ONNX_URL", "").strip()
    if not url:
        raise RuntimeError(
            f"AdaFace model not found: {_MODEL_PATH}\n"
            "Copy the file there, or set env ADAFACE_ONNX_URL to download it on startup "
            "(see README, model files section)."
        )
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MODEL_PATH.with_name(_MODEL_PATH.name + ".part")
    logger.info("Downloading AdaFace model from %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "javstar-face/1.0"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        milestone = 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done >> 25 > milestone:  # log every 32 MiB
                milestone = done >> 25
                logger.info("  %d MiB / %s MiB", done >> 20, total >> 20 if total else "?")
    expected = os.environ.get("ADAFACE_SHA256", "").strip().lower()
    if expected:
        h = hashlib.sha256()
        with open(tmp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            tmp.unlink()
            raise RuntimeError(f"AdaFace model sha256 mismatch: {h.hexdigest()} != {expected}")
        logger.info("AdaFace model sha256 verified")
    tmp.replace(_MODEL_PATH)
    logger.info("AdaFace model saved to %s", _MODEL_PATH)


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                if not _MODEL_PATH.is_file():
                    _download_model()
                options = ort.SessionOptions()
                options.enable_cpu_mem_arena = False
                options.log_severity_level = 3
                _session = ort.InferenceSession(
                    str(_MODEL_PATH), options, providers=["CPUExecutionProvider"]
                )
                logger.info("AdaFace loaded from %s", _MODEL_PATH)
    return _session


def warmup() -> None:
    """Load the ONNX session eagerly so a missing model fails at startup."""
    _get_session()


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
