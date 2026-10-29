# AI RAW Image Processor

Turns camera RAW files into finished JPEGs. AI picks the color grade and tone for each photo,
AI super-resolution upscales it, and the export carries detailed metadata.

- **Any RAW**: DNG, Apple ProRAW (including iCloud's `.RAW-IMAGE` exports), CR2/CR3, NEF, ARW,
  RAF, ORF, RW2, PEF, SRW and the other ~1000 cameras LibRaw supports. Format is detected from
  file content, so odd extensions are fine.
- **AI grading**: a scene classifier (CLIP) picks a style prior, then the editing parameters are
  optimized by gradient ascent against an ensemble of models trained on human aesthetic and
  quality ratings (LAION-Aesthetics, TOPIQ-AVA, TOPIQ-KonIQ). Photographic guard rails stop it
  from overdoing the edit.
- **AI upscaling**: Real-ESRGAN 2× or 4×, tiled for 8 GB GPUs, blended with Lanczos so textures
  stay photographic.
- **Detailed metadata**: original EXIF + GPS, AI caption/title/keywords, star rating, offline
  reverse-geocoded location, and the full edit recipe and scores, written to EXIF, XMP and IPTC.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Model weights (~3 GB) download automatically on first run. They are cached in
`~/.cache/torch/hub`, `~/.cache/huggingface` and `./models`.

## Usage

```powershell
# a folder (recursive), output to ./output, 2x AI upscale (default)
.\.venv\Scripts\python process.py "..\icloud photos" -o output

# single files, no upscaling, vivid look, side-by-side comparison sheets, JSON sidecars
.\.venv\Scripts\python process.py IMG_0741.RAW-IMAGE photo.CR3 --scale 1 --style vivid --compare --sidecar

# credit + copyright in every file, cap size at 6000 px, remove GPS
.\.venv\Scripts\python process.py raws -o out --artist "Jane Doe" --copyright "(c) 2026 Jane Doe" --max-size 6000 --strip-gps
```

| Option | Default | Meaning |
|---|---|---|
| `--scale {1,2,4}` | 2 | Real-ESRGAN upscale factor (1 = off, light sharpening only) |
| `--texture-blend` | 0.3 | share of Lanczos mixed into the AI upscale |
| `--style {natural,vivid,moody}` | natural | look the AI optimizes within |
| `--steps` | 60 | optimization steps per image (more = slower, slightly better) |
| `--quality` | 95 | JPEG quality (always 4:4:4, progressive, optimized) |
| `--max-size` | – | cap the long edge in pixels |
| `--artist`, `--copyright` | – | written to EXIF, XMP and IPTC |
| `--no-caption`, `--no-geocode`, `--strip-gps` | – | metadata controls |
| `--sidecar` | off | per-image `.json` with parameters, scores, caption and keywords |
| `--compare` | off | `OUT/_compare/*.jpg`: LibRaw-default vs AI side by side |
| `--overwrite` | off | re-process files that already have an output |

Each run also writes `OUT/_report.json`, a summary of every image (scene, scores before and
after, parameters, timing) plus any failures.

## How it works

```
RAW ─► LibRaw decode (linear Rec.2020, as-shot WB, BaselineExposure, clip-level detection)
    ─► CLIP scene analysis ─► style prior (bounds + start point; e.g. night stays dark)
    ─► AI colorist: Adam over 13 parameters on a 512 px preview, maximizing
         aesthetic(LAION + TOPIQ-AVA) + quality(TOPIQ-KonIQ) − guard rails
    ─► full-resolution render on GPU (same differentiable pipeline)
    ─► Real-ESRGAN upscale ─► BLIP captions re-ranked by CLIP + CLIP keywords
    ─► JPEG (sRGB ICC) + EXIF/XMP/IPTC
```

**Pipeline** ([rawai/pipeline.py](rawai/pipeline.py)) is written in PyTorch so it is
differentiable: white balance (temp/tint) → exposure → local tone mapping (highlights/shadows
on an edge-aware guided-filter base layer, clarity on its detail) → gamut compression →
filmic log-sigmoid tone curve (contrast/whites) → blacks → vibrance (skin-protected) /
saturation → split toning. Sensor-clipped light sources (moon, lamps) are excluded from
highlight recovery so they stay bright instead of turning grey.

**Guard rails** ([rawai/grader.py](rawai/grader.py)):
- a prior penalty keeps edits near the scene's start point, weighted heavier for white balance;
- penalties for clipped highlights, crushed shadows, mean brightness outside the scene's band,
  and excess colorfulness;
- random crops and flips each step;
- the result is the average of the parameters over the second half of the run (Polyak), and is
  kept only if it beats the start point on a clean evaluation.

## Metadata written

| Where | What |
|---|---|
| EXIF | Make, Model, Lens, exposure (shutter, aperture, ISO, bias, program, metering, flash, WB), focal length + 35 mm equiv., DateTimeOriginal + sub-seconds + time-zone offsets, **full GPS block**; Orientation=1, ColorSpace=sRGB, pixel dimensions, Software, ImageDescription (caption), Artist/Copyright |
| XMP | `dc:title`, `dc:description`, `dc:subject` keywords, `xmp:Rating` (1–5 from aesthetic score), `xmp:Label` (scene), CreateDate/DateCreated with offset, `photoshop:City/State/Country`, `Iptc4xmpCore:CountryCode`, `xmpMM` provenance (source file, document IDs), `aux:Lens`, ICC name; custom `rawai:` namespace with scene + confidence, aesthetic/quality scores before and after, models used, upscale model, source firmware/dimensions, and every edit parameter (plus a JSON recipe) |
| IPTC | Caption, Headline, ObjectName, Keywords, City, Province/State, Country + ISO code, Date/Time created, Program/version, By-line, Copyright (UTF-8) |
| ICC | sRGB IEC61966-2.1 |

Location names come from an offline GeoNames lookup, so no coordinates leave the machine. Adobe
`crs:` develop settings are deliberately not written: Lightroom would re-apply them on top
of the already-graded pixels.
