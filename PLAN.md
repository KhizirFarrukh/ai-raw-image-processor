# AI RAW Image Processor — Project Plan

## Goal
A command-line tool that takes **any camera RAW** (DNG, Apple ProRAW / iCloud `.RAW-IMAGE`,
CR2/CR3, NEF, ARW, RAF, ORF, RW2, PEF, SRW, …), uses AI to find the best color grade, tone,
and edits for that specific photo, optionally AI-upscales it, and exports a high-quality JPEG
with rich, accurate metadata.

## What we're working with (findings)
- Test set: `../icloud photos/` — 59 files, 2.0 GB, all `.RAW-IMAGE` (iCloud's export name).
  - Header `MM.*…APPLEDNG` → they are DNGs. Cameras: iPhone 13 Pro, iPhone 15 Pro Max
    (Apple ProRAW — *linear*, already demosaiced, 4032×3024) and realme 9 Pro+ (Bayer DNG, 4096×3072).
  - LibRaw 0.22 (via `rawpy`) decodes all tested files in ~0.7 s. LibRaw detects format by
    content, so the odd extension is irrelevant and ~1000 camera models are covered.
- Hardware: RTX 4060 Laptop (8 GB VRAM), Ryzen 5 7640HS, 15 GB RAM, Python 3.12.

## Architecture

```
rawai/
  decode.py     RAW → linear, scene-referred float32 RGB (wide-gamut working space)
  pipeline.py   differentiable (PyTorch) editing pipeline driven by ~14 global parameters
  scene.py      CLIP zero-shot scene analysis → style prior, parameter bounds, keywords
  scorers.py    AI judges: LAION aesthetic (CLIP ViT-L/14) + TOPIQ/MUSIQ image quality
  grader.py     the "AI colorist": heuristic init → gradient-ascent on AI scores w/ regularizers
  upscale.py    AI super-resolution (Real-ESRGAN via spandrel), tiled, blended to limit hallucination
  caption.py    BLIP image captioning → title/description
  metadata.py   full EXIF/XMP/IPTC transfer + AI-generated metadata (pyexiv2)
  export.py     sRGB conversion, ICC embedding, JPEG encoding
  cli.py        batch CLI, progress, per-image JSON sidecar / report
process.py      entry point
```

### 1. Decode (`decode.py`)
- `rawpy.postprocess(gamma=(1,1), no_auto_bright=True, output_bps=16, use_camera_wb=True,
  output_color=Rec2020 if available else ProPhoto)` → linear float32 in a wide gamut so
  saturated colors aren't clipped before grading.
- Read `BaselineExposure` (Apple ProRAW relies heavily on it) and apply it as the starting EV.
- Highlight mode "blend" for Bayer files to recover clipped highlights.
- LibRaw applies the orientation flip, so output pixels are upright (EXIF Orientation becomes 1).

### 2. Differentiable editing pipeline (`pipeline.py`)
All operations are PyTorch ops, so the *same* code renders a 512 px preview with gradients
(for the AI search) and the full-resolution image on the GPU (no gradients).
Order (linear light → display):
1. **White balance** — temperature and tint as channel gains around the as-shot WB.
2. **Exposure** — EV multiplier.
3. **Highlights / Shadows** — local tone mapping in log-luminance using a blurred base layer
   (edge-aware-ish via multi-scale blur), so skies recover and shadows open without halos.
4. **Filmic tone curve** — log-space sigmoid with contrast, whites and blacks controls and a
   soft highlight shoulder (raw highlights roll off instead of clipping).
5. **Color** — saturation and vibrance (vibrance weighted toward low-saturation pixels and
   protects skin hues), plus a split-tone (shadow/highlight warmth) for "grading".
6. **Clarity** — large-radius local contrast on luminance.
7. **Dehaze-lite** — black-level lift compensation driven by the dark-channel estimate.
8. Wide gamut → sRGB with soft gamut compression, then sRGB OETF.
Each parameter is passed through `tanh` to stay inside scene-dependent bounds.

### 3. Scene understanding (`scene.py`)
- CLIP zero-shot over prompts: portrait, landscape, cityscape, night, sunset, food,
  indoor, snow, beach, forest, architecture, pet, macro, concert/low light…
- The top scene selects a **style prior**: e.g. night → keep it dark (target luminance band low),
  portrait → gentle contrast plus skin-protecting vibrance, landscape → more clarity/vibrance,
  sunset → keep warmth instead of neutralizing WB.
- The same CLIP pass yields **keywords** for metadata.

### 4. AI judges (`scorers.py`)
- **LAION aesthetic predictor** (CLIP ViT-L/14 + MLP; trained on human aesthetic ratings).
- **TOPIQ-NR** and **MUSIQ** no-reference quality models (trained on human quality opinions).
- Loaded via `pyiqa` in differentiable (`as_loss`) mode, using a weighted ensemble so no single
  model's bias (e.g. "more saturation is always better") dominates.

### 5. AI grader (`grader.py`) — the core
1. **Heuristic init**: auto-exposure to a scene-dependent log-average key, gray-world/white-patch
   WB hint blended with the as-shot WB, highlights set from the clipping risk, shadows from the
   dark-pixel ratio.
2. **Optimization**: Adam over the ~14 parameters for ~80 steps on a 512 px preview, maximizing
   `w·aesthetic + w·quality` minus regularizers:
   - stay close to the heuristic/prior (prevents adversarial "score hacking"),
   - clipped-highlight and crushed-shadow penalties,
   - luminance band per scene (night stays night),
   - colorfulness ceiling and a skin-hue preservation term.
3. Keep the best-scoring iterate (not just the last one). Record before/after scores.
Expected cost: ~5–10 s per image on the RTX 4060.

### 6. Scaling (`upscale.py`)
- `--scale 1|2|4` (default 2): Real-ESRGAN (x2plus / x4plus weights) via `spandrel`, tiled with
  overlap in fp16 to fit 8 GB. The output is blended with a Lanczos upscale (default 30 %) to
  keep natural texture and avoid the "painted" look. Then a light luminance-only output sharpen.
- `--max-size` long-edge cap; at scale 1 the full-resolution graded image is exported as-is.

### 7. Metadata (`metadata.py`) — detailed, accurate
Source tags are read from the RAW with Exiv2 (pyexiv2) and written into the JPEG:
- **EXIF**: Make, Model, LensMake/LensModel, DateTimeOriginal/Digitized + OffsetTime (time zone),
  SubSecTime, ExposureTime, FNumber, ISO, FocalLength, FocalLengthIn35mm, ExposureBias,
  ExposureProgram, MeteringMode, Flash, WhiteBalance, BrightnessValue, SceneType, etc.
- **GPS**: latitude/longitude/altitude/direction/speed/timestamp, carried over unchanged.
- **Fixed-up tags**: Orientation=1 (pixels are already upright), PixelX/YDimension = new size,
  ColorSpace=sRGB, Software = "AI RAW Processor x.y", ModifyDate = now. DNG-only tags,
  raw strip offsets, and the embedded raw preview/thumbnail are dropped.
- **XMP**: `dc:title` and `dc:description` (AI caption), `dc:subject` (CLIP keywords plus scene),
  `xmp:Rating` (1–5 from the aesthetic score), `xmp:CreatorTool`, `photoshop:DateCreated`,
  `xmpMM` provenance (original filename). *(Revised during build: Camera Raw `crs:` settings
  are NOT written. Lightroom would re-apply them on top of the already-graded pixels, so the
  recipe lives only in the custom namespace below.)*
- Custom `rawai:` XMP namespace: scene label and confidence, aesthetic/quality scores
  before and after, upscale model, and the exact pipeline parameters (reproducible).
- **IPTC**: Caption-Abstract, Keywords, ObjectName — for apps that only read IPTC.
- **ICC**: an embedded sRGB profile.
- A `.json` sidecar per image (optional) with the full parameter set and scores.

### 8. Export (`export.py`)
- Pillow JPEG, quality 95 (configurable), 4:4:4 chroma (no subsampling) for colorful edges,
  optimized Huffman tables, progressive. Output name `IMG_0741.jpg` next to an `--out` folder.

### 9. CLI (`process.py`)
```
python process.py INPUT [INPUT ...] -o OUT_DIR
     [--scale 2] [--quality 95] [--max-size N] [--steps 80] [--style auto|natural|vivid|moody]
     [--no-caption] [--sidecar] [--device cuda|cpu] [--compare]
```
- Accepts files, folders (recursive) and globs; skips non-raw files with a warning.
- Models are loaded once per batch; per-file failures are logged without stopping the batch.
- `--compare` also writes a side-by-side "naive LibRaw auto vs AI" preview for evaluation.

## Dependencies
torch + torchvision (CUDA 12.8), rawpy, numpy, pillow, pyiqa (aesthetic/IQA models),
spandrel (SR model loader), transformers (BLIP captioner), open_clip/CLIP (via pyiqa), pyexiv2,
opencv-python-headless, tqdm. Model weights download automatically on first run and are cached.

## Validation plan
1. Unit-ish checks: decode every test file; confirm orientation; pipeline identity params ≈ neutral.
2. Run the full batch on all 59 test images.
3. Quantitative: mean AI-score gain vs the naive LibRaw auto-bright render.
4. Qualitative: visually inspect a sample across cameras/scenes (day, night, indoor, portrait)
   for color casts, halos, clipping, over-saturation, SR artifacts.
5. Verify metadata: read the JPEGs back with Exiv2 to confirm EXIF, GPS, XMP and IPTC are present and correct.
6. Tune priors/regularizer weights based on what goes wrong, then re-run.
