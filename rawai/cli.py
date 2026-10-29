"""Command-line batch processor: RAW in -> AI-graded, upscaled, metadata-rich JPEG out."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from fractions import Fraction

import cv2
import numpy as np
import torch

from . import SOFTWARE_NAME, metadata
from .decode import RawImage, decode, is_raw_path
from .export import resize_long_edge, save_jpeg, sharpen, to_uint8
from .grader import Grader, to_preview
from .pipeline import PARAM_SPEC, render, render_full
from .scene import SCENES

SCENE_MIN_CONF = 0.35
CAPTION_MIN_SIM = 0.25 # CLIP ViT-B/16 cosine similarity below which a caption is distrusted
DEFAULT_MODELS_DIR =os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def collect_inputs(inputs: list[str]) -> list[str]:
    files: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, names in os.walk(item):
                files += [os.path.join(root, n) for n in sorted(names) if is_raw_path(n)]
        elif any(c in item for c in "*?["):
            files += sorted(glob.glob(item, recursive=True))
        elif os.path.isfile(item):
            files.append(item)          # explicit files are attempted whatever the extension
        else:
            print(f"warning: not found: {item}", file=sys.stderr)
    seen, unique = set(), []
    for f in files:
        key = os.path.normcase(os.path.abspath(f))
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def _num(s: str | None) -> float | None:
    try:
        return float(Fraction(s.split()[0])) if s else None
    except (ValueError, ZeroDivisionError):
        return None


def shooting_summary(exif: dict) -> str:
    parts = []
    lens_mm, eq = _num(exif.get("Exif.Photo.FocalLength")), exif.get("Exif.Photo.FocalLengthIn35mmFilm")
    if lens_mm:
        parts.append(f"{lens_mm:.1f} mm" + (f" ({eq} mm equiv.)" if eq else ""))
    if (fn := _num(exif.get("Exif.Photo.FNumber"))):
        parts.append(f"f/{fn:.1f}")
    if (t := _num(exif.get("Exif.Photo.ExposureTime"))):
        parts.append(f"{t:g} s" if t >= 1 else f"1/{round(1 / t)} s")
    if (iso := exif.get("Exif.Photo.ISOSpeedRatings")):
        parts.append(f"ISO {iso.split()[0]}")
    return ", ".join(parts)


def describe(caption: str | None, raw: RawImage, scene: str, location: dict | None) -> tuple[str, str]:
    title = caption or f"{scene.title()} photo"
    sentence = f"{title.rstrip('.')}."
    shot = shooting_summary(raw.exif)
    detail = f" Shot on {raw.camera}" + (f", {shot}" if shot else "")
    if location:
        detail += f", near {location['city']}, {location['state']}, {location['country']}"
    return title, sentence + detail + "."


def naive_render(path: str) -> np.ndarray:
    """What a plain RAW converter gives (LibRaw defaults, auto-brightness) - for comparison."""
    import rawpy
    with rawpy.imread(path) as r:
        return r.postprocess(use_camera_wb=True, half_size=True).astype(np.float32) / 255.0


def save_compare(path: str, naive: np.ndarray, ai: np.ndarray, height: int = 900):
    def fit(im):
        k = height / im.shape[0]
        return cv2.resize(im, (round(im.shape[1] * k), height), interpolation=cv2.INTER_AREA)
    a, b = fit(naive), fit(ai)
    gap = np.ones((height, 12, 3), np.float32)
    sheet = to_uint8(np.concatenate([a, gap, b], axis=1))
    for text, x in (("LibRaw default", 14), ("AI graded", a.shape[1] + 26)):
        cv2.putText(sheet, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(sheet, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def render_full_safe(lin: np.ndarray, params: dict, device: str, clip_level: float) -> np.ndarray:
    try:
        return render_full(lin, params, device, clip_level)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return render_full(lin, params, "cpu", clip_level)
    finally:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="process.py",
        description="AI RAW processor: decodes any camera RAW (DNG, ProRAW/.RAW-IMAGE, CR2/CR3, "
                    "NEF, ARW, RAF, ORF, RW2, ...), finds the best grade with AI, upscales, "
                    "and exports a JPEG with detailed metadata.")
    ap.add_argument("inputs", nargs="+", help="RAW files, folders (recursive) or glob patterns")
    ap.add_argument("-o", "--out", default="output", help="output folder (default: ./output)")
    ap.add_argument("--scale", type=int, choices=(1, 2, 4), default=2,
                    help="AI super-resolution factor (Real-ESRGAN). 1 = no upscaling. Default 2")
    ap.add_argument("--texture-blend", type=float, default=0.3,
                    help="share of plain Lanczos mixed into the AI upscale (0-1, default 0.3)")
    ap.add_argument("--max-size", type=int, default=None, help="cap the long edge (pixels)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--style", choices=("natural", "vivid", "moody"), default="natural",
                    help="grading style the AI optimizes within (default natural)")
    ap.add_argument("--steps", type=int, default=60, help="AI optimization steps per image")
    ap.add_argument("--artist", help="creator name written to EXIF/XMP/IPTC")
    ap.add_argument("--copyright", help="copyright notice written to EXIF/XMP/IPTC")
    ap.add_argument("--no-caption", action="store_true", help="skip AI captioning")
    ap.add_argument("--no-geocode", action="store_true", help="don't add city/country from GPS")
    ap.add_argument("--strip-gps", action="store_true", help="remove GPS coordinates from output")
    ap.add_argument("--sidecar", action="store_true", help="write a .json sidecar per image")
    ap.add_argument("--compare", action="store_true",
                    help="also write LibRaw-default vs AI side-by-side sheets to OUT/_compare")
    ap.add_argument("--overwrite", action="store_true", help="re-process existing outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    warnings.filterwarnings("ignore")
    files = collect_inputs(args.inputs)
    if not files:
        print("No RAW files found.", file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    dev = args.device
    print(f"{SOFTWARE_NAME} | {len(files)} file(s) | device: "
          f"{torch.cuda.get_device_name(0) if dev.startswith('cuda') else 'cpu'}")

    from .scene import SceneAnalyzer
    from .scorers import Judges
    print("Loading AI models (first run downloads weights)...", flush=True)
    judges = Judges(dev)
    grader = Grader(judges, dev, steps=args.steps)
    analyzer = SceneAnalyzer(dev)
    captioner = None
    if not args.no_caption:
        from .caption import Captioner
        captioner = Captioner(dev)
    upscaler = None
    if args.scale > 1:
        from .upscale import Upscaler
        upscaler = Upscaler(args.scale, dev, args.models_dir)

    report, failures = [], []
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        out_path = os.path.join(args.out, os.path.splitext(name)[0] + ".jpg")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{i}/{len(files)}] {name}: exists, skipping (use --overwrite)")
            continue
        t0 = time.time()
        try:
            raw = decode(path)
            src_h, src_w = raw.rgb.shape[:2]

            # Scene understanding on a quick auto-exposed preview.
            prev = to_preview(raw.rgb, 512, dev)
            neutral = {k: torch.tensor(v[2], device=dev) for k, v in PARAM_SPEC.items()}
            lum = prev.mean(1).flatten().clamp_min(1e-6)
            neutral["exposure"] = torch.log2(0.16 / torch.exp(torch.log(lum).mean()))
            with torch.no_grad():
                quick = render(prev, neutral, raw.clip_level).squeeze(0).permute(1, 2, 0).cpu().numpy()
            prior, conf, scene_dist = analyzer.classify(to_uint8(quick))
            if conf < SCENE_MIN_CONF:       # unsure -> don't impose a specific scene's look
                prior = SCENES[-1]

            res = grader.grade(raw.rgb, raw.baseline_ev, prior, args.style, raw.clip_level)
            img = render_full_safe(raw.rgb, res.params, dev, raw.clip_level)

            if upscaler:
                img = upscaler(img, texture_blend=args.texture_blend)
            else:
                img = sharpen(img, 0.3)
            img = resize_long_edge(img, args.max_size)
            out_h, out_w = img.shape[:2]

            thumb = to_uint8(resize_long_edge(img, 768))
            concepts = analyzer.keywords(thumb)
            caption, caption_sim = None, None
            if captioner:
                caption, caption_sim = analyzer.best_caption(thumb, captioner.candidates(thumb))
                if caption_sim < CAPTION_MIN_SIM:   # no candidate actually matches the image
                    caption = f"{prior.name.title()} scene" if prior.name != "general" else "Photo"
            location = None if args.no_geocode else metadata.reverse_geocode(raw.exif)
            keywords = ([prior.name] if prior.name != "general" else []) + \
                [k for k in concepts if k != prior.name]
            if location:
                keywords += [location["city"], location["country"]]
            title, description = describe(caption, raw, prior.name, location)

            exif = dict(raw.exif)
            if args.strip_gps:
                exif = {k: v for k, v in exif.items() if not k.startswith("Exif.GPSInfo.")}
                location = None
            info = {
                "exif": exif, "source_name": name, "camera": raw.camera,
                "width": out_w, "height": out_h, "src_width": src_w, "src_height": src_h,
                "title": title, "caption": description if caption else None,
                "keywords": list(dict.fromkeys(keywords)), "location": location,
                "scene": prior.name, "scene_confidence": conf, "style": args.style,
                "params": res.params, "scores_before": res.scores_before,
                "scores_after": res.scores_after, "scoring_models": judges.description,
                "steps": args.steps, "scale": args.scale,
                "upscale_model": upscaler.name if upscaler else None,
                "rating": rating(res.scores_after["aesthetic"]),
                "artist": args.artist, "copyright": args.copyright,
            }
            size = save_jpeg(img, out_path, info, quality=args.quality)
            del img

            record = {
                "file": name, "output": out_path, "camera": raw.camera, "scene": prior.name,
                "scene_confidence": round(conf, 3), "scene_candidates": scene_dist,
                "title": title, "description": description, "caption_clip_similarity": caption_sim,
                "keywords": info["keywords"],
                "location": location, "rating": info["rating"],
                "output_size": [out_w, out_h], "bytes": size,
                "params": {k: round(v, 4) for k, v in res.params.items()},
                "init_params": {k: round(v, 4) for k, v in res.init_params.items()},
                "scores_before": res.scores_before, "scores_after": res.scores_after,
                "ai_search_improved_on_start": res.improved,
                "seconds": round(time.time() - t0, 1),
            }
            report.append(record)
            if args.sidecar:
                with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as fh:
                    json.dump(record, fh, indent=2, ensure_ascii=False)
            if args.compare:
                os.makedirs(os.path.join(args.out, "_compare"), exist_ok=True)
                save_compare(os.path.join(args.out, "_compare", os.path.splitext(name)[0] + ".jpg"),
                             naive_render(path), np.asarray(thumb, np.float32) / 255.0)
            sb, sa = res.scores_before, res.scores_after
            print(f"[{i}/{len(files)}] {name} -> {os.path.basename(out_path)} | {prior.name} "
                  f"({conf:.0%}) | aesthetic {sb['aesthetic']:.2f}->{sa['aesthetic']:.2f} "
                  f"quality {sb['quality']:.2f}->{sa['quality']:.2f} | {out_w}x{out_h} "
                  f"{size / 1e6:.1f} MB | {time.time() - t0:.1f}s", flush=True)
            if caption:
                print(f"      \"{title}\" (clip {caption_sim:.2f})  [{', '.join(info['keywords'][:8])}]")
        except Exception as e:  # keep the batch going
            failures.append({"file": name, "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(files)}] {name}: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()

    write_report(os.path.join(args.out, "_report.json"), args, report, failures)
    if report:
        gain_a = np.mean([r["scores_after"]["aesthetic"] - r["scores_before"]["aesthetic"] for r in report])
        gain_q = np.mean([r["scores_after"]["quality"] - r["scores_before"]["quality"] for r in report])
        print(f"\nDone: {len(report)} exported, {len(failures)} failed. "
              f"Mean AI score gain vs plain conversion: aesthetic {gain_a:+.2f}, quality {gain_q:+.3f}")
    return 0 if not failures else 2


def write_report(path: str, args, report: list[dict], failures: list[dict]):
    """Merge this run into the folder's report so re-running a few files keeps the rest."""
    images, old_failures = {}, []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
            images = {r["file"]: r for r in old.get("images", [])}
            old_failures = old.get("failures", [])
        except (OSError, ValueError):
            pass
    images.update({r["file"]: r for r in report})
    done = {r["file"] for r in report}
    failures = [f for f in old_failures if f["file"] not in done | {g["file"] for g in failures}] + failures
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"software": SOFTWARE_NAME, "settings": dict(vars(args)),
                   "images": sorted(images.values(), key=lambda r: r["file"]),
                   "failures": failures}, fh, indent=2, ensure_ascii=False)


def rating(aesthetic: float) -> int:
    """Map the aesthetic ensemble score onto 1-5 stars. Everyday photos land around 4.3-5.5 on
    the models' 1-10 scale: <4.3 = 1 star, 4.3-4.6 = 2, 4.6-4.9 = 3, 4.9-5.2 = 4, >=5.2 = 5."""
    return int(np.clip(1 + (aesthetic - 4.0) // 0.3, 1, 5))
