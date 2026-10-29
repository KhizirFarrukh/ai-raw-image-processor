"""AI judges: an ensemble of differentiable aesthetic and image-quality models (via pyiqa).

  laion_aes  CLIP ViT-L/14 + MLP trained on human aesthetic ratings (LAION-Aesthetics), 1-10
  topiq_iaa  TOPIQ trained on AVA photo-contest aesthetic votes, 1-10
  topiq_nr   TOPIQ trained on KonIQ-10k human quality opinions (exposure, noise, blur...), 0-1

Using several models trained on different human-opinion datasets keeps the grade from
exploiting any single model's quirks.
"""

from __future__ import annotations

import warnings

import torch

AESTHETIC_MODELS = ("laion_aes", "topiq_iaa")
QUALITY_MODEL = "topiq_nr"


class Judges:
    def __init__(self, device: str):
        import pyiqa

        self.device = device
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.metrics = {n: pyiqa.create_metric(n, as_loss=True, device=device)
                            for n in (*AESTHETIC_MODELS, QUALITY_MODEL)}
        for m in self.metrics.values():
            m.eval()
            for prm in m.parameters():
                prm.requires_grad_(False)

    @property
    def description(self) -> str:
        return "+".join(self.metrics)

    def raw_scores(self, img: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            return {n: m(img).float().mean() for n, m in self.metrics.items()}

    @staticmethod
    def combine(s: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> (objective, aesthetic 1-10, quality 0-1). One objective unit ~ a clearly better photo."""
        aesthetic = sum(s[n] for n in AESTHETIC_MODELS) / len(AESTHETIC_MODELS)
        quality = s[QUALITY_MODEL]
        return aesthetic + 4.0 * quality, aesthetic, quality

    @torch.no_grad()
    def evaluate(self, img: torch.Tensor) -> dict[str, float]:
        s = self.raw_scores(img)
        _, aes, q = self.combine(s)
        return {"aesthetic": float(aes), "quality": float(q), **{k: float(v) for k, v in s.items()}}
