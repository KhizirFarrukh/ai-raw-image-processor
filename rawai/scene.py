"""CLIP zero-shot scene understanding: picks a grading style prior and produces keywords."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

# Each scene: CLIP prompts + the grading prior used by the AI colorist.
#   key        target log-average luminance (scene "brightness intent")
#   lum_band   allowed mean display luminance (sRGB-encoded) of the result
#   bounds     per-parameter (lo, hi) overrides for the search space
#   init       starting values on top of the neutral defaults
#   crush_ok   fraction of near-black pixels that is acceptable
@dataclass
class ScenePrior:
    name: str
    prompts: list[str]
    key: float = 0.16
    lum_band: tuple[float, float] = (0.36, 0.58)
    bounds: dict = field(default_factory=dict)
    init: dict = field(default_factory=dict)
    crush_ok: float = 0.01


_COMMON_BOUNDS = {
    "temp": (-0.4, 0.4), "tint": (-0.25, 0.25), "highlights": (-1.0, 0.3), "shadows": (-0.2, 0.8),
    "contrast": (-0.5, 0.6), "whites": (-0.6, 0.6), "blacks": (-0.8, 0.4), "clarity": (-0.3, 0.5),
    "vibrance": (-0.3, 0.6), "saturation": (-0.4, 0.3), "split_shadows": (-0.6, 0.6),
    "split_highlights": (-0.6, 0.6),
}
_COMMON_INIT = {"highlights": -0.3, "shadows": 0.2, "contrast": 0.1, "clarity": 0.1, "vibrance": 0.15}

SCENES = [
    ScenePrior("portrait", ["a portrait photo of a person", "a close-up photo of a face",
                            "a selfie", "a photo of a group of people"],
               bounds={"clarity": (-0.4, 0.15), "saturation": (-0.4, 0.15), "contrast": (-0.5, 0.35)},
               init={"clarity": 0.0, "vibrance": 0.1}),
    ScenePrior("landscape", ["a landscape photo of nature", "a photo of mountains",
                             "a photo of a green field and trees", "a scenic view of the countryside"],
               bounds={"clarity": (-0.2, 0.7), "vibrance": (-0.2, 0.75)},
               init={"clarity": 0.25, "vibrance": 0.25}),
    ScenePrior("nature", ["a close-up photo of a flower", "a photo of plants and leaves",
                          "a macro photo of a plant", "a photo of a blooming flower"],
               bounds={"clarity": (-0.2, 0.5), "vibrance": (-0.2, 0.7)},
               init={"clarity": 0.15, "vibrance": 0.25}),
    ScenePrior("cityscape", ["a photo of a city street", "a photo of buildings and architecture",
                             "an urban scene in daylight", "a photo of a road with cars"],
               bounds={"clarity": (-0.2, 0.6)}, init={"clarity": 0.2}),
    ScenePrior("night", ["a photo taken at night", "a dark night street with lights",
                         "a night sky", "a dimly lit scene at night"],
               key=0.07, lum_band=(0.14, 0.38), crush_ok=0.08,
               bounds={"shadows": (-0.3, 0.5), "highlights": (-1.0, 0.0), "clarity": (-0.3, 0.35)},
               init={"shadows": 0.1, "highlights": -0.3, "contrast": 0.15}),
    ScenePrior("sunset", ["a photo of a sunset", "a golden hour photo", "a sunrise over the horizon",
                          "an orange evening sky"],
               key=0.11, lum_band=(0.25, 0.5), crush_ok=0.04,
               bounds={"temp": (-0.1, 0.6)}, init={"temp": 0.05, "highlights": -0.5, "vibrance": 0.25}),
    ScenePrior("indoor", ["a photo of an indoor room", "an interior of a house",
                          "a photo inside a restaurant", "a photo of an office"],
               key=0.14, bounds={"shadows": (-0.1, 0.8)}, init={"shadows": 0.3}),
    ScenePrior("food", ["a photo of food", "a plate of food on a table", "a photo of a dessert",
                        "a photo of a drink"],
               bounds={"temp": (-0.2, 0.5), "vibrance": (-0.1, 0.7)}, init={"vibrance": 0.3, "temp": 0.05}),
    ScenePrior("beach", ["a photo of a beach", "a photo of the sea and waves", "a tropical coast"],
               key=0.19, lum_band=(0.42, 0.64), bounds={"vibrance": (-0.2, 0.7)}, init={"vibrance": 0.25}),
    ScenePrior("snow", ["a snowy landscape", "a photo of snow", "a winter scene with snow"],
               key=0.26, lum_band=(0.5, 0.72), init={"vibrance": 0.1}),
    ScenePrior("animal", ["a photo of a dog", "a photo of a cat", "a photo of a pet",
                          "a photo of a wild animal", "a photo of a bird"],
               bounds={"clarity": (-0.2, 0.4)}),
    ScenePrior("event", ["a photo of a concert", "a party with colorful lights",
                         "a stage with spotlights", "a wedding celebration"],
               key=0.09, lum_band=(0.2, 0.45), crush_ok=0.06, init={"highlights": -0.5}),
    ScenePrior("sky", ["a photo of the sky and clouds", "a photo of the moon", "a dramatic cloudy sky"],
               key=0.14, crush_ok=0.03, bounds={"clarity": (-0.2, 0.6)}, init={"highlights": -0.5}),
    ScenePrior("general", ["a photo", "a snapshot", "an everyday photo"]),
]

KEYWORDS = [
    "sky", "clouds", "sunset", "sunrise", "night", "moon", "stars", "city", "street", "road",
    "building", "skyscraper", "architecture", "house", "bridge", "mosque", "church", "temple",
    "monument", "car", "traffic", "motorcycle", "bus", "train", "airplane", "boat", "ship", "harbor",
    "person", "people", "crowd", "child", "family", "friends", "selfie", "portrait", "man", "woman",
    "dog", "cat", "bird", "horse", "animal", "tree", "trees", "forest", "palm tree", "flowers",
    "garden", "park", "grass", "field", "mountain", "hills", "desert", "sand", "beach", "sea",
    "ocean", "waves", "lake", "river", "waterfall", "snow", "rain", "fog", "food", "dessert",
    "coffee", "drink", "restaurant", "cafe", "table", "kitchen", "room", "interior", "bedroom",
    "office", "shop", "market", "mall", "lights", "neon", "fireworks", "concert", "stage", "party",
    "wedding", "festival", "sport", "stadium", "computer", "phone", "book", "art", "painting",
    "sculpture", "window", "door", "wall", "stairs", "reflection", "silhouette", "shadow",
    "golden hour", "blue hour", "landscape", "cityscape", "skyline", "rooftop", "aerial view",
    "close-up", "macro", "text", "sign", "graffiti", "vehicle", "airport", "highway", "pool",
    "playground", "cricket", "football",
]


class SceneAnalyzer:
    MODEL = "openai/clip-vit-base-patch16"

    def __init__(self, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(self.MODEL).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(self.MODEL)
        self._scene_index = [(i, p) for i, s in enumerate(SCENES) for p in s.prompts]
        self.scene_emb = self._text([p for _, p in self._scene_index])
        self.kw_emb = self._text([f"a photo of {k}" for k in KEYWORDS])

    @staticmethod
    def _normalize(e):
        # transformers >= 5 returns a model output object instead of a bare tensor
        e = e if torch.is_tensor(e) else e.pooler_output
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _text(self, texts):
        inp = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
        return self._normalize(self.model.get_text_features(**inp))

    @torch.no_grad()
    def _image(self, img_u8: np.ndarray):
        inp = self.proc(images=img_u8, return_tensors="pt").to(self.device)
        return self._normalize(self.model.get_image_features(**inp))

    def classify(self, img_u8: np.ndarray) -> tuple[ScenePrior, float, dict[str, float]]:
        img = self._image(img_u8)
        logits = (100.0 * img @ self.scene_emb.T).squeeze(0)
        probs = logits.softmax(-1).cpu().numpy()
        per_scene = np.zeros(len(SCENES))
        for (si, _), pr in zip(self._scene_index, probs):
            per_scene[si] += pr
        best = int(per_scene.argmax())
        dist = {SCENES[i].name: float(per_scene[i]) for i in np.argsort(-per_scene)[:4]}
        return SCENES[best], float(per_scene[best]), dist

    def keywords(self, img_u8: np.ndarray, k: int = 8, margin: float = 0.025) -> list[str]:
        """Concepts scoring within `margin` cosine similarity of the best match."""
        img = self._image(img_u8)
        sims = (img @ self.kw_emb.T).squeeze(0).cpu().numpy()
        order = np.argsort(-sims)[:k]
        return [KEYWORDS[i] for i in order if sims[i] >= sims[order[0]] - margin]

    def best_caption(self, img_u8: np.ndarray, candidates: list[str]) -> tuple[str, float]:
        """Re-rank caption candidates by CLIP image-text similarity (guards against hallucination)."""
        sims = (self._image(img_u8) @ self._text(candidates).T).squeeze(0).cpu().numpy()
        i = int(sims.argmax())
        return candidates[i], float(sims[i])


def prior_bounds(prior: ScenePrior, style: str) -> tuple[dict, dict]:
    bounds = dict(_COMMON_BOUNDS, **prior.bounds)
    init = dict(_COMMON_INIT, **prior.init)
    if style == "vivid":
        bounds["vibrance"] = (bounds["vibrance"][0], bounds["vibrance"][1] + 0.25)
        bounds["saturation"] = (bounds["saturation"][0], bounds["saturation"][1] + 0.2)
        bounds["contrast"] = (bounds["contrast"][0], bounds["contrast"][1] + 0.2)
        init["vibrance"] = init.get("vibrance", 0) + 0.15
        init["contrast"] = init.get("contrast", 0) + 0.1
    elif style == "moody":
        bounds["saturation"] = (-0.6, 0.1)
        bounds["blacks"] = (-0.8, 0.8)
        init.update(contrast=init.get("contrast", 0) + 0.15, saturation=-0.2,
                    split_shadows=-0.3, split_highlights=0.25)
    return bounds, init
