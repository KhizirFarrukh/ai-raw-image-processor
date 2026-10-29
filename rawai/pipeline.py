"""Differentiable photo-editing pipeline (PyTorch).

Input:  linear Rec.2020 scene-referred RGB, shape (1, 3, H, W).
Output: display-referred, sRGB-encoded RGB in [0, 1], shape (1, 3, H, W).

Every operation is a smooth tensor op, so the same function renders a small preview with
gradients (for the AI search over parameters) and the full-resolution image without them.
The local tone-mapping base layer is computed at a fixed low resolution and upsampled
(fast guided filter), which keeps preview and full-res renders visually consistent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# name -> (default lower bound, default upper bound, neutral value)
PARAM_SPEC: dict[str, tuple[float, float, float]] = {
    "exposure":   (-4.0, 6.0, 0.0),   # EV, absolute (includes the camera's baseline exposure)
    "temp":       (-1.0, 1.0, 0.0),   # + warmer / - cooler, relative to as-shot WB
    "tint":       (-1.0, 1.0, 0.0),   # + magenta / - green
    "highlights": (-1.0, 1.0, 0.0),   # - recovers bright areas
    "shadows":    (-1.0, 1.0, 0.0),   # + opens dark areas
    "contrast":   (-1.0, 1.0, 0.0),
    "whites":     (-1.0, 1.0, 0.0),   # + moves the white point down (brighter whites)
    "blacks":     (-1.0, 1.0, 0.0),   # - deepens blacks / + lifts (matte)
    "clarity":    (-1.0, 1.0, 0.0),   # large-scale local contrast
    "vibrance":   (-1.0, 1.0, 0.0),   # saturation boost weighted to muted, non-skin colors
    "saturation": (-1.0, 1.0, 0.0),
    "split_shadows":    (-1.0, 1.0, 0.0),  # shadow warmth (- teal / + amber)
    "split_highlights": (-1.0, 1.0, 0.0),  # highlight warmth
}
PARAM_NAMES = list(PARAM_SPEC)

MID_GREY = 0.18
_LOG2_GREY = math.log2(MID_GREY)

# Linear Rec.2020 -> linear sRGB (D65)
_REC2020_TO_SRGB = torch.tensor([
    [1.660491, -0.587641, -0.072850],
    [-0.124550, 1.132900, -0.008349],
    [-0.018151, -0.100579, 1.118730],
])
_LUMA_2020 = (0.2627, 0.6780, 0.0593)
_LUMA_709 = (0.2126, 0.7152, 0.0722)


def neutral_params(device="cpu") -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v[2], device=device) for k, v in PARAM_SPEC.items()}


def luminance(x: torch.Tensor, weights=_LUMA_709) -> torch.Tensor:
    w = torch.tensor(weights, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def srgb_encode(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-8) ** (1 / 2.4) - 0.055)


def srgb_decode(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp(0.0, 1.0)
    return torch.where(y <= 0.04045, y / 12.92, ((y + 0.055) / 1.055) ** 2.4)


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    return F.avg_pool2d(x, 2 * r + 1, stride=1, padding=r, count_include_pad=False)


def guided_base(log_lum: torch.Tensor, short_side: int = 256, radius: int = 8,
                eps: float = 0.5) -> torch.Tensor:
    """Edge-aware smooth base layer of log-luminance (fast guided filter, self-guided)."""
    h, w = log_lum.shape[-2:]
    scale = short_side / min(h, w)
    lo = F.interpolate(log_lum, size=(max(1, round(h * scale)), max(1, round(w * scale))),
                       mode="bilinear", align_corners=False, antialias=True)
    mean_i = _box(lo, radius)
    var_i = _box(lo * lo, radius) - mean_i * mean_i
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    a = F.interpolate(_box(a, radius), size=(h, w), mode="bilinear", align_corners=False)
    b = F.interpolate(_box(b, radius), size=(h, w), mode="bilinear", align_corners=False)
    return a * log_lum + b


def _white_balance(x, temp, tint):
    gains = torch.stack([torch.exp(0.35 * temp), torch.exp(-0.25 * tint), torch.exp(-0.35 * temp)])
    w = torch.tensor(_LUMA_2020, device=x.device, dtype=x.dtype)
    gains = gains / (gains * w).sum()          # keep luminance unchanged
    return x * gains.view(1, 3, 1, 1)


def _local_tone(x, p, sensor_clip):
    lum = luminance(x, _LUMA_2020).clamp_min(1e-6)
    log_l = torch.log2(lum)
    base = guided_base(log_l)
    detail = log_l - base
    rel = base - _LOG2_GREY                    # stops relative to middle grey
    w_sh = torch.sigmoid((-rel - 1.0) * 1.5)
    # Sensor-clipped pixels (moon, lamps, sun) hold no detail to recover: pulling them down
    # only turns light sources grey, so highlight recovery skips them.
    w_hi = torch.sigmoid((rel - 1.0) * 1.5) * (1.0 - sensor_clip)
    base = base + 1.6 * p["shadows"] * w_sh + 1.6 * p["highlights"] * w_hi
    new_log_l = base + detail * (1.0 + 0.9 * p["clarity"])
    return x * torch.exp2(new_log_l - log_l)


def _gamut_compress(x):
    """Pull out-of-gamut (negative) components toward the pixel's luminance, keeping hue."""
    y = luminance(x).clamp_min(1e-6)
    m = x.min(1, keepdim=True).values
    k = torch.where(m < 0, y / (y - m), torch.ones_like(m)).clamp(0.0, 1.0)
    return y + (x - y) * k


def _filmic(v, contrast, whites):
    """Log-space sigmoid tone curve -> sRGB-encoded display value in [0, 1]."""
    s = 0.7 * torch.exp(0.45 * contrast)
    c = math.log(1 / 0.42 - 1) / s             # grey maps to ~0.42 before normalisation
    x_white = 3.2 - 1.3 * whites
    x_black = -7.5
    f = lambda t: torch.sigmoid(s * (t - c))   # noqa: E731
    stops = torch.log2(v.clamp_min(1e-7) / MID_GREY)
    lo, hi = f(torch.as_tensor(x_black)), f(x_white)
    return ((f(stops) - lo) / (hi - lo)).clamp(0.0, 1.0)


def _tone_curve(x, p):
    per_channel = _filmic(x, p["contrast"], p["whites"])
    lum = luminance(x).clamp_min(1e-7)
    lum_lin = srgb_decode(_filmic(lum, p["contrast"], p["whites"]))
    hue_preserving = srgb_encode(x * (lum_lin / lum))
    # Per-channel curves desaturate highlights naturally; luminance curves preserve hue.
    return 0.5 * per_channel + 0.5 * hue_preserving


def _blacks(y, blacks):
    crush = 0.06 * F.relu(-blacks)
    lift = 0.05 * F.relu(blacks)
    y = ((y - crush) / (1 - crush)).clamp(0.0, 1.0)
    return y * (1 - lift) + lift


def skin_weight(y: torch.Tensor) -> torch.Tensor:
    lum = luminance(y)
    cb, cr = y[:, 2:3] - lum, y[:, 0:1] - lum
    angle = torch.atan2(cr, cb)
    chroma = torch.sqrt(cb * cb + cr * cr + 1e-8)
    return torch.exp(-((angle - 2.3) / 0.35) ** 2) * torch.sigmoid((chroma - 0.02) * 60)


def _color(y, p):
    lum = luminance(y)
    chroma = y - lum
    sat = y.max(1, keepdim=True).values - y.min(1, keepdim=True).values
    vib_w = (1.0 - sat).clamp(0, 1) ** 2 * (1.0 - 0.7 * skin_weight(y))
    gain = 1.0 + 0.5 * p["saturation"] + 0.7 * p["vibrance"] * vib_w
    y = lum + chroma * gain.clamp_min(0.0)
    warm = torch.tensor([1.0, 0.15, -1.0], device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    w_sh = (1.0 - lum).clamp(0, 1) ** 2
    w_hi = lum.clamp(0, 1) ** 2
    y = y + 0.04 * warm * (p["split_shadows"] * w_sh + p["split_highlights"] * w_hi)
    return y.clamp(0.0, 1.0)


def render(lin_rec2020: torch.Tensor, p: dict[str, torch.Tensor],
           clip_level: float = float("inf")) -> torch.Tensor:
    if math.isfinite(clip_level):
        peak = lin_rec2020.max(1, keepdim=True).values / clip_level
        sensor_clip = torch.sigmoid((peak - 0.95) * 80)
    else:
        sensor_clip = torch.zeros_like(lin_rec2020[:, :1])
    x = lin_rec2020 * torch.exp2(p["exposure"])
    x = _white_balance(x, p["temp"], p["tint"])
    x = _local_tone(x, p, sensor_clip)
    x = torch.einsum("ij,bjhw->bihw", _REC2020_TO_SRGB.to(x.device, x.dtype), x)
    x = _gamut_compress(x)
    y = _tone_curve(x, p)
    # Clipped sources are the brightest thing in the scene; keep them near white, not grey.
    y = y + (1.0 - y) * (0.7 * sensor_clip)
    y = _blacks(y, p["blacks"])
    return _color(y, p)


def render_full(lin_rec2020_np, p: dict[str, float], device: str, clip_level: float = float("inf")):
    """Render a full-resolution numpy HxWx3 image without gradients. Returns float32 HxWx3."""
    with torch.no_grad():
        t = torch.from_numpy(lin_rec2020_np).permute(2, 0, 1).unsqueeze(0).to(device)
        pt = {k: torch.tensor(float(v), device=device) for k, v in p.items()}
        out = render(t, pt, clip_level)
        return out.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
