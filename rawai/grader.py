"""The AI colorist: searches the editing-parameter space for the grade the AI judges prefer.

1. Heuristic start: auto-exposure to the scene's target key, highlight protection from the
   brightest pixels, and the scene/style prior's starting look.
2. Gradient ascent (Adam) through the differentiable pipeline and the judge ensemble on a
   small preview, with random crops/flips each step (robustness against score hacking).
3. Photographic guard rails as penalties: stay near the prior, don't clip highlights or crush
   shadows beyond what the scene allows, keep brightness in the scene's band (night stays
   night), and cap colorfulness.
4. The parameters are Polyak-averaged over the second half of the run and accepted only if
   they beat the starting grade on a clean (non-augmented) evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .pipeline import PARAM_NAMES, PARAM_SPEC, luminance, render
from .scene import ScenePrior, prior_bounds
from .scorers import Judges

# White balance was measured by the camera; aesthetic models like warm, magenta-ish casts, so
# departing from the as-shot WB costs more than other edits.
PRIOR_WEIGHT = {"temp": 4.0, "tint": 6.0}
COLORFULNESS_CAP ={"natural": 0.36, "vivid": 0.46, "moody": 0.28}


@dataclass
class GradeResult:
    params: dict[str, float]
    init_params: dict[str, float]
    scores_before: dict[str, float]
    scores_after: dict[str, float]
    objective_before: float
    objective_after: float
    improved: bool


def colorfulness(y: torch.Tensor) -> torch.Tensor:
    """Hasler & Suesstrunk colorfulness on sRGB-encoded [0,1] values."""
    rg = y[:, 0] - y[:, 1]
    yb = 0.5 * (y[:, 0] + y[:, 1]) - y[:, 2]
    std = torch.sqrt(rg.var() + yb.var() + 1e-8)
    mean = torch.sqrt(rg.mean() ** 2 + yb.mean() ** 2 + 1e-8)
    return std + 0.3 * mean


def to_preview(lin: np.ndarray, long_side: int, device: str) -> torch.Tensor:
    h, w = lin.shape[:2]
    k = long_side / max(h, w)
    small = cv2.resize(lin, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).to(device)


class Grader:
    def __init__(self, judges: Judges, device: str, steps: int = 60, preview: int = 512,
                 lr: float = 0.1, seed: int = 0):
        self.judges, self.device, self.steps, self.preview, self.lr = judges, device, steps, preview, lr
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    # ---------- start point ----------
    def initial(self, lin: torch.Tensor, baseline_ev: float, prior: ScenePrior, style: str):
        bounds, init = prior_bounds(prior, style)
        lum = luminance(lin, (0.2627, 0.6780, 0.0593)).flatten().clamp_min(1e-6)
        lo_q, hi_q = torch.quantile(lum, torch.tensor([0.01, 0.99], device=lum.device))
        core = lum[(lum >= lo_q) & (lum <= hi_q)]
        log_avg = torch.exp(torch.log(core).mean()).item()
        ev = float(np.log2(prior.key / log_avg))
        ev = float(np.clip(ev, baseline_ev - 2.5, baseline_ev + 3.5))
        # Protect highlights: how many stops above middle grey is the 99.5th percentile?
        p995 = torch.quantile(lum[:: max(1, lum.numel() // 200_000)], 0.995).item() * 2 ** ev
        over = np.log2(max(p995, 1e-6) / 0.18) - 3.0
        if over > 0:
            # Capped: a hard pull turns small bright sources (moon, lamps) a dull grey.
            init["highlights"] = min(init.get("highlights", 0.0), max(-0.6, -0.3 - 0.15 * over))
        p0 = {k: v[2] for k, v in PARAM_SPEC.items()}
        p0.update(init)
        p0["exposure"] = ev
        bounds["exposure"] = (ev - 1.5, ev + 1.5)
        for k, (lo, hi) in bounds.items():
            p0[k] = float(np.clip(p0[k], lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo)))
        return p0, bounds

    # ---------- objective ----------
    def _penalty(self, y, p_vec, p0_vec, span, prior: ScenePrior, cf_cap):
        lum = luminance(y)
        w = torch.tensor([PRIOR_WEIGHT.get(k, 1.0) for k in PARAM_NAMES], device=p_vec.device)
        pen = 0.5 * (w * ((p_vec - p0_vec) / span) ** 2).sum()
        clip_frac = torch.sigmoid((y.max(1).values - 0.99) * 300).mean()
        pen = pen + 30 * F.relu(clip_frac - 0.004)
        crush_frac = torch.sigmoid((0.015 - lum) * 300).mean()
        pen = pen + 30 * F.relu(crush_frac - prior.crush_ok)
        m = lum.mean()
        lo, hi = prior.lum_band
        pen = pen + 100 * (F.relu(lo - m) ** 2 + F.relu(m - hi) ** 2)
        pen = pen + 60 * F.relu(colorfulness(y) - cf_cap) ** 2
        return pen

    def _augment(self, y):
        _, _, h, w = y.shape
        s = 0.8 + 0.2 * torch.rand(1, generator=self.gen).item()
        ch, cw = int(h * s), int(w * s)
        oy = int(torch.randint(0, h - ch + 1, (1,), generator=self.gen))
        ox = int(torch.randint(0, w - cw + 1, (1,), generator=self.gen))
        y = y[:, :, oy:oy + ch, ox:ox + cw]
        return y.flip(-1) if torch.rand(1, generator=self.gen).item() < 0.5 else y

    def _render(self, lin, pd):
        return render(lin, pd, self._clip_level)

    def _evaluate(self, lin, pd, p_vec, p0_vec, span, prior, cf_cap):
        with torch.no_grad():
            y = self._render(lin, pd)
            s = self.judges.raw_scores(y)
            obj, aes, q = self.judges.combine(s)
            total = obj - self._penalty(y, p_vec, p0_vec, span, prior, cf_cap)
            scores = {"aesthetic": float(aes), "quality": float(q), **{k: float(v) for k, v in s.items()}}
        return float(total), scores

    # ---------- main ----------
    def grade(self, lin_np: np.ndarray, baseline_ev: float, prior: ScenePrior, style: str,
              clip_level: float = float("inf")) -> GradeResult:
        dev = self.device
        self._clip_level = clip_level
        lin = to_preview(lin_np, self.preview, dev)
        p0, bounds = self.initial(lin, baseline_ev, prior, style)
        lo = torch.tensor([bounds.get(k, PARAM_SPEC[k][:2])[0] for k in PARAM_NAMES], device=dev)
        hi = torch.tensor([bounds.get(k, PARAM_SPEC[k][:2])[1] for k in PARAM_NAMES], device=dev)
        span = hi - lo
        p0_vec = torch.tensor([p0[k] for k in PARAM_NAMES], device=dev)
        as_dict = lambda v: dict(zip(PARAM_NAMES, v))  # noqa: E731
        cf_cap = COLORFULNESS_CAP.get(style, COLORFULNESS_CAP["natural"])

        # "Before" = plain conversion at the camera's own baseline exposure, no edits.
        neutral = {k: torch.tensor(v[2], device=dev) for k, v in PARAM_SPEC.items()}
        neutral["exposure"] = torch.tensor(float(baseline_ev), device=dev)
        with torch.no_grad():
            scores_before = self.judges.evaluate(self._render(lin, neutral))
            cf_cap = max(cf_cap, float(colorfulness(self._render(lin, as_dict(p0_vec)))))
        obj_init, _ = self._evaluate(lin, as_dict(p0_vec), p0_vec, p0_vec, span, prior, cf_cap)

        u = torch.logit(((p0_vec - lo) / span).clamp(1e-3, 1 - 1e-3)).clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=self.lr)
        ema, n_ema = torch.zeros_like(p0_vec), 0
        for step in range(self.steps):
            p_vec = lo + span * torch.sigmoid(u)
            y = self._render(lin, as_dict(p_vec))
            obj, _, _ = self.judges.combine(self.judges.raw_scores(self._augment(y)))
            loss = -obj + self._penalty(y, p_vec, p0_vec, span, prior, cf_cap)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step >= self.steps // 2:
                ema += p_vec.detach()
                n_ema += 1

        final_vec = ema / max(n_ema, 1)
        obj_final, scores_after = self._evaluate(lin, as_dict(final_vec), final_vec, p0_vec,
                                                 span, prior, cf_cap)
        improved = obj_final > obj_init
        if not improved:
            final_vec = p0_vec
            obj_final, scores_after = self._evaluate(lin, as_dict(p0_vec), p0_vec, p0_vec,
                                                     span, prior, cf_cap)
        return GradeResult(
            params={k: float(v) for k, v in zip(PARAM_NAMES, final_vec)},
            init_params=p0, scores_before=scores_before, scores_after=scores_after,
            objective_before=obj_init, objective_after=obj_final, improved=improved,
        )
