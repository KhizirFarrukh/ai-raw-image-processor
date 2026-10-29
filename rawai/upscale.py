"""AI super-resolution with Real-ESRGAN (loaded via spandrel), tiled to fit in GPU memory."""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
MODELS = {
    2: ("RealESRGAN_x2plus.pth", f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth"),
    4: ("RealESRGAN_x4plus.pth", f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth"),
}


class Upscaler:
    def __init__(self, scale: int, device: str, models_dir: str, tile: int = 512, overlap: int = 32):
        import spandrel

        name, url = MODELS[scale]
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            os.makedirs(models_dir, exist_ok=True)
            torch.hub.download_url_to_file(url, path, progress=True)
        desc = spandrel.ModelLoader().load_from_file(path)
        self.half = device.startswith("cuda") and desc.supports_half
        self.model = desc.model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        self.name = name.rsplit(".", 1)[0]
        self.scale, self.device, self.tile, self.overlap = scale, device, tile, overlap

    @torch.no_grad()
    def _run(self, patch: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(self.device)
        t = t.half() if self.half else t
        return self.model(t).float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()

    def __call__(self, img: np.ndarray, texture_blend: float = 0.3) -> np.ndarray:
        """img: HxWx3 float32 sRGB-encoded in [0,1]. Returns (H*s)x(W*s)x3 float32.

        The GAN output is blended with a Lanczos upscale: GAN super-resolution can paint
        plausible-but-fake texture, and mixing back some of the faithful interpolation keeps
        foliage, skin and noise looking photographic.
        """
        h, w, _ = img.shape
        s, t, o = self.scale, self.tile, self.overlap
        out = np.empty((h * s, w * s, 3), np.float32)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - o, 0), max(x0 - o, 0)
                py1, px1 = min(y1 + o, h), min(x1 + o, w)
                sr = self._run(np.ascontiguousarray(img[py0:py1, px0:px1]))
                out[y0 * s:y1 * s, x0 * s:x1 * s] = sr[(y0 - py0) * s:(y1 - py0) * s,
                                                       (x0 - px0) * s:(x1 - px0) * s]
        if texture_blend > 0:
            lanczos = cv2.resize(img, (w * s, h * s), interpolation=cv2.INTER_LANCZOS4)
            out *= 1.0 - texture_blend
            out += texture_blend * lanczos
        return np.clip(out, 0.0, 1.0, out=out)
