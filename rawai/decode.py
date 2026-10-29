"""RAW decoding: any LibRaw-supported file -> linear, scene-referred float32 RGB (Rec.2020)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import pyexiv2
import rawpy

# Extensions we pick up when scanning folders. LibRaw detects the format from file content,
# so anything else can still be passed explicitly.
RAW_EXTENSIONS = {
    ".dng", ".raw-image", ".raw", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".srf",
    ".sr2", ".raf", ".orf", ".rw2", ".pef", ".ptx", ".srw", ".x3f", ".3fr", ".fff", ".iiq",
    ".mos", ".mef", ".mrw", ".erf", ".kdc", ".dcr", ".rwl", ".gpr", ".k25", ".cap", ".eip",
}


@dataclass
class RawImage:
    path: str
    rgb: np.ndarray                 # HxWx3 float32, linear Rec.2020, 1.0 = sensor white
    baseline_ev: float              # DNG BaselineExposure (camera's intended brightness offset)
    clip_level: float = float("inf")  # value at which the sensor clipped (inf = nothing clipped)
    exif: dict = field(default_factory=dict)
    xmp: dict = field(default_factory=dict)
    iptc: dict = field(default_factory=dict)

    @property
    def camera(self) -> str:
        make = self.exif.get("Exif.Image.Make", "").strip()
        model = self.exif.get("Exif.Image.Model", "").strip()
        return model if model.lower().startswith(make.lower()) else f"{make} {model}".strip()


def is_raw_path(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def _rational(value: str | None, default: float = 0.0) -> float:
    if not value:
        return default
    try:
        return float(Fraction(value.split()[0]))
    except (ValueError, ZeroDivisionError):
        return default


def read_metadata(path: str) -> tuple[dict, dict, dict]:
    """Read EXIF/XMP/IPTC with Exiv2. Never fails: missing metadata just yields empty dicts."""
    try:
        with open(path, "rb") as fh:
            img = pyexiv2.ImageData(fh.read())
        try:
            exif, xmp, iptc = img.read_exif(), img.read_xmp(), img.read_iptc()
        finally:
            img.close()
        return exif, xmp, iptc
    except Exception:
        return {}, {}, {}


def decode(path: str) -> RawImage:
    exif, xmp, iptc = read_metadata(path)
    with rawpy.imread(path) as raw:
        rgb16 = raw.postprocess(
            output_color=rawpy.ColorSpace.Rec2020,
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            highlight_mode=rawpy.HighlightMode.Blend,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    rgb = rgb16.astype(np.float32) * (1.0 / 65535.0)
    baseline_ev = _rational(exif.get("Exif.Image.BaselineExposure"), 0.0)
    return RawImage(path=path, rgb=rgb, baseline_ev=baseline_ev, clip_level=detect_clip_level(rgb),
                    exif=exif, xmp=xmp, iptc=iptc)


def detect_clip_level(rgb: np.ndarray, min_fraction: float = 1e-4) -> float:
    """Sensor clipping shows up as a plateau of pixels sharing the maximum value. LibRaw's
    white-balance scaling puts it below 1.0, so it is measured rather than assumed."""
    peak = rgb.max(axis=2)[::2, ::2]
    top = float(peak.max())
    if top <= 0 or (peak >= top * 0.995).mean() < min_fraction:
        return float("inf")
    return top
