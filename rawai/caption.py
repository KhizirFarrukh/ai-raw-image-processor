"""BLIP image captioning -> title and description for metadata."""

from __future__ import annotations

import re

import numpy as np
import torch

# BLIP sometimes emits made-up tokens learned from noisy alt-text.
_JUNK = re.compile(r"\b(arafed|araffe|arafe|araf|there is|there are)\b\s*", re.IGNORECASE)


class Captioner:
    MODEL = "Salesforce/blip-image-captioning-large"

    def __init__(self, device: str):
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.proc = BlipProcessor.from_pretrained(self.MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.MODEL, dtype=self.dtype).to(device).eval()

    @torch.no_grad()
    def candidates(self, img_u8: np.ndarray, n: int = 3) -> list[str]:
        """Several captions (unconditional + photo-prompted beams) for CLIP re-ranking."""
        texts = []
        for prompt in (None, "a photograph of"):
            inp = self.proc(images=img_u8, text=prompt, return_tensors="pt").to(self.device, self.dtype)
            out = self.model.generate(**inp, num_beams=n + 1, num_return_sequences=n,
                                      max_new_tokens=40, min_new_tokens=6, repetition_penalty=1.3)
            texts += self.proc.batch_decode(out, skip_special_tokens=True)
        cleaned = [clean(t) for t in texts]
        return list(dict.fromkeys(c for c in cleaned if len(c.split()) >= 3))


def clean(text: str) -> str:
    text = re.sub(r"^a photograph of\s+", "", text.strip(), flags=re.IGNORECASE)
    text = _JUNK.sub("", text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(a|an) (a|an) ", r"\1 ", text)
    return text[:1].upper() + text[1:] if text else text
