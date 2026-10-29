"""Final finishing and JPEG encoding (sRGB ICC profile, 4:4:4, metadata)."""

from __future__ import annotations

import io
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageCms

from . import metadata

_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def resize_long_edge(img: np.ndarray, max_size: int | None) -> np.ndarray:
    h, w = img.shape[:2]
    if not max_size or max(h, w) <= max_size:
        return img
    k = max_size / max(h, w)
    return cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)


def sharpen(img: np.ndarray, amount: float = 0.35, sigma: float = 1.0) -> np.ndarray:
    """Luminance-only unsharp mask (avoids colored halos)."""
    if amount <= 0:
        return img
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    detail = lum - cv2.GaussianBlur(lum, (0, 0), sigma)
    return np.clip(img + amount * detail[..., None], 0.0, 1.0)


def to_uint8(img: np.ndarray, seed: int = 0) -> np.ndarray:
    """Quantize with light triangular dither so smooth skies don't band."""
    rng = np.random.default_rng(seed)
    noise = (rng.random(img.shape[:2], np.float32) - rng.random(img.shape[:2], np.float32))[..., None]
    return np.clip(img * 255.0 + 0.5 + noise * 0.5, 0, 255).astype(np.uint8)


def save_jpeg(img: np.ndarray, path: str, info: dict, quality: int = 95) -> int:
    buf = io.BytesIO()
    Image.fromarray(to_uint8(img)).save(
        buf, "JPEG", quality=quality, subsampling=0, optimize=True, progressive=True,
        icc_profile=_SRGB_ICC,
    )
    data = metadata.apply(buf.getvalue(), info)
    write_atomic(path, data)
    return len(data)


def write_atomic(path: str, data: bytes, attempts: int = 5):
    """Write via a temp file + rename, retrying while Windows has the target locked (photo
    viewers, Explorer preview and the search indexer briefly hold JPEGs open)."""
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if i == attempts - 1:
                os.remove(tmp)
                raise
            time.sleep(1.0 + i)
