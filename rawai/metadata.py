"""Metadata for exported JPEGs: carry over EXIF/GPS from the RAW, add AI-generated XMP/IPTC.

Written with Exiv2 (pyexiv2):
  EXIF  camera, lens, exposure, dates + time zone, GPS (copied); orientation/size/color space
        fixed up for the rendered pixels; caption, software, artist/copyright.
  XMP   dc:title/description/subject, xmp:Rating, dates, location (photoshop:/Iptc4xmpCore:),
        provenance (xmpMM), and the AI edit recipe + scores in a custom `rawai:` namespace.
  IPTC  caption, headline, keywords, location, dates, program - for IPTC-only readers.

Adobe `crs:` develop settings are deliberately NOT written: Lightroom/ACR would apply them
again on top of the already-graded pixels.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import uuid
from fractions import Fraction

import pyexiv2

from . import SOFTWARE_NAME, __version__

RAWAI_NS = "http://ns.ai-raw-processor.local/rawai/1.0/"
pyexiv2.registerNs(RAWAI_NS, "rawai")

# Exif.Image (IFD0) tags worth keeping; everything else in IFD0 of a DNG is raw-specific.
_IFD0_KEEP = {
    "Exif.Image.Make", "Exif.Image.Model", "Exif.Image.Artist", "Exif.Image.Copyright",
    "Exif.Image.ImageDescription", "Exif.Image.HostComputer", "Exif.Image.CameraSerialNumber",
}
# Exif.Photo tags that describe the original file/encoding rather than the capture.
_PHOTO_DROP = {
    "Exif.Photo.MakerNote", "Exif.Photo.PixelXDimension", "Exif.Photo.PixelYDimension",
    "Exif.Photo.ColorSpace", "Exif.Photo.ComponentsConfiguration",
    "Exif.Photo.CompressedBitsPerPixel", "Exif.Photo.InteroperabilityTag",
    "Exif.Photo.ExifVersion", "Exif.Photo.FlashpixVersion",
    # pixel coordinates in the un-rotated, un-scaled source - wrong for the exported image
    "Exif.Photo.SubjectArea", "Exif.Photo.SubjectLocation",
}


def _ratio(s: str) -> float:
    return float(Fraction(s))


_RATIONAL = re.compile(r"^-?\d+/\d+(\s+-?\d+/\d+)*$")


def _reduce_rationals(value: str) -> str:
    """Some cameras write rationals like 3750000000/125000000 that overflow Exiv2's signed
    32-bit parsing and get dropped; reduce them (30/1) and bound the denominator."""
    if not _RATIONAL.match(value.strip()):
        return value
    out = []
    for part in value.split():
        num, den = part.split("/")
        if int(den) == 0:
            out.append(part)
            continue
        f = Fraction(int(num), int(den))
        if abs(f.numerator) > 2**31 - 1 or f.denominator > 2**31 - 1:
            f = f.limit_denominator(1_000_000)
        out.append(f"{f.numerator}/{f.denominator}")
    return " ".join(out)


def gps_decimal(exif: dict) -> tuple[float, float] | None:
    try:
        def dms(key, ref_key, neg):
            d, m, s = (_ratio(v) for v in exif[key].split())
            val = d + m / 60 + s / 3600
            return -val if exif.get(ref_key, "").strip().upper() == neg else val
        lat = dms("Exif.GPSInfo.GPSLatitude", "Exif.GPSInfo.GPSLatitudeRef", "S")
        lon = dms("Exif.GPSInfo.GPSLongitude", "Exif.GPSInfo.GPSLongitudeRef", "W")
        return lat, lon
    except (KeyError, ValueError, ZeroDivisionError):
        return None


def reverse_geocode(exif: dict) -> dict | None:
    """Offline lookup (GeoNames cities >1000 people) - no network, no location leaves the PC."""
    coords = gps_decimal(exif)
    if coords is None:
        return None
    try:
        import pycountry
        import reverse_geocoder as rg
        hit = rg.search([coords], mode=1, verbose=False)[0]
        country = pycountry.countries.get(alpha_2=hit["cc"])
        return {
            "city": hit["name"], "state": hit["admin1"], "country_code": hit["cc"],
            "country": country.name if country else hit["cc"],
            "lat": coords[0], "lon": coords[1],
        }
    except Exception:
        return None


def capture_datetime(exif: dict) -> tuple[str | None, str | None]:
    """Return (ISO-8601 capture time with sub-seconds and offset, EXIF offset string)."""
    raw = exif.get("Exif.Photo.DateTimeOriginal") or exif.get("Exif.Image.DateTimeOriginal") \
        or exif.get("Exif.Image.DateTime")
    if not raw:
        return None, None
    try:
        t = dt.datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None, None
    iso = t.strftime("%Y-%m-%dT%H:%M:%S")
    sub = (exif.get("Exif.Photo.SubSecTimeOriginal") or "").strip()
    if sub.isdigit():
        iso += "." + sub
    offset = (exif.get("Exif.Photo.OffsetTimeOriginal") or exif.get("Exif.Photo.OffsetTime") or "").strip()
    return iso + offset, offset or None


def build_exif(src: dict, width: int, height: int, caption: str | None,
               artist: str | None, copyright_: str | None) -> dict:
    out = {}
    for k, v in src.items():
        if not v.strip():
            continue
        if k in _IFD0_KEEP or k.startswith("Exif.GPSInfo."):
            out[k] = _reduce_rationals(v)
        elif k.startswith("Exif.Photo.") and k not in _PHOTO_DROP:
            out[k] = _reduce_rationals(v)
    # Some phones (e.g. realme) store DateTimeOriginal in IFD0; EXIF readers expect it in Photo.
    if "Exif.Photo.DateTimeOriginal" not in out and "Exif.Image.DateTimeOriginal" in src:
        out["Exif.Photo.DateTimeOriginal"] = src["Exif.Image.DateTimeOriginal"]
    now = dt.datetime.now().astimezone()
    out.update({
        "Exif.Image.Orientation": "1",              # pixels are already upright
        "Exif.Image.Software": SOFTWARE_NAME,
        "Exif.Image.DateTime": now.strftime("%Y:%m:%d %H:%M:%S"),
        "Exif.Image.XResolution": "300/1",
        "Exif.Image.YResolution": "300/1",
        "Exif.Image.ResolutionUnit": "2",
        "Exif.Photo.ExifVersion": "48 50 51 50",     # "0232"
        "Exif.Photo.ColorSpace": "1",               # sRGB
        "Exif.Photo.PixelXDimension": str(width),
        "Exif.Photo.PixelYDimension": str(height),
        "Exif.Photo.OffsetTime": now.strftime("%z")[:3] + ":" + now.strftime("%z")[3:],
    })
    if caption:
        out["Exif.Image.ImageDescription"] = caption.encode("ascii", "replace").decode()
    if artist:
        out["Exif.Image.Artist"] = artist
    if copyright_:
        out["Exif.Image.Copyright"] = copyright_
    return out


def _lang(text: str) -> dict:
    return {'lang="x-default"': text}


def build_xmp(info: dict) -> dict:
    """`info` is the per-image result dict assembled by the CLI."""
    src_exif = info["exif"]
    now_iso = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    created, _ = capture_datetime(src_exif)
    keywords = info.get("keywords", [])
    xmp = {
        "Xmp.xmp.CreatorTool": SOFTWARE_NAME,
        "Xmp.xmp.ModifyDate": now_iso,
        "Xmp.xmp.MetadataDate": now_iso,
        "Xmp.xmp.Rating": str(info["rating"]),
        "Xmp.xmp.Label": info["scene"].title(),
        "Xmp.dc.format": "image/jpeg",
        "Xmp.dc.subject": keywords,
        "Xmp.dc.source": info["source_name"],
        "Xmp.xmpMM.PreservedFileName": info["source_name"],
        "Xmp.xmpMM.DocumentID": f"xmp.did:{uuid.uuid4()}",
        "Xmp.xmpMM.InstanceID": f"xmp.iid:{uuid.uuid4()}",
        "Xmp.photoshop.ColorMode": "3",
        "Xmp.photoshop.ICCProfile": "sRGB IEC61966-2.1",
        "Xmp.tiff.Orientation": "1",
        "Xmp.exif.PixelXDimension": str(info["width"]),
        "Xmp.exif.PixelYDimension": str(info["height"]),
    }
    if created:
        xmp["Xmp.xmp.CreateDate"] = created
        xmp["Xmp.photoshop.DateCreated"] = created
        xmp["Xmp.exif.DateTimeOriginal"] = created
    if info.get("caption"):
        xmp["Xmp.dc.title"] = _lang(info["title"])
        xmp["Xmp.dc.description"] = _lang(info["caption"])
        xmp["Xmp.photoshop.Headline"] = info["title"]
    if src_exif.get("Exif.Photo.LensModel"):
        xmp["Xmp.aux.Lens"] = src_exif["Exif.Photo.LensModel"]
    if info.get("artist"):
        xmp["Xmp.dc.creator"] = [info["artist"]]
        xmp["Xmp.photoshop.Credit"] = info["artist"]
    if info.get("copyright"):
        xmp["Xmp.dc.rights"] = _lang(info["copyright"])
    loc = info.get("location")
    if loc:
        xmp.update({
            "Xmp.photoshop.City": loc["city"],
            "Xmp.photoshop.State": loc["state"],
            "Xmp.photoshop.Country": loc["country"],
            "Xmp.Iptc4xmpCore.CountryCode": loc["country_code"],
        })

    p = info["params"]
    xmp.update({
        "Xmp.rawai.Version": __version__,
        "Xmp.rawai.SourceFile": info["source_name"],
        "Xmp.rawai.SourceCamera": info["camera"],
        "Xmp.rawai.SourceFirmware": src_exif.get("Exif.Image.Software", "").strip(),
        "Xmp.rawai.Scene": info["scene"],
        "Xmp.rawai.SceneConfidence": f"{info['scene_confidence']:.3f}",
        "Xmp.rawai.Style": info["style"],
        "Xmp.rawai.AestheticScoreBefore": f"{info['scores_before']['aesthetic']:.3f}",
        "Xmp.rawai.AestheticScoreAfter": f"{info['scores_after']['aesthetic']:.3f}",
        "Xmp.rawai.QualityScoreBefore": f"{info['scores_before']['quality']:.3f}",
        "Xmp.rawai.QualityScoreAfter": f"{info['scores_after']['quality']:.3f}",
        "Xmp.rawai.ScoringModels": info["scoring_models"],
        "Xmp.rawai.OptimizationSteps": str(info["steps"]),
        "Xmp.rawai.UpscaleFactor": str(info["scale"]),
        "Xmp.rawai.UpscaleModel": info.get("upscale_model") or "none",
        "Xmp.rawai.SourceDimensions": f"{info['src_width']}x{info['src_height']}",
        "Xmp.rawai.EditRecipe": json.dumps({k: round(float(v), 4) for k, v in p.items()}),
    })
    for k, v in p.items():
        tag = "".join(part.title() for part in k.split("_"))
        xmp[f"Xmp.rawai.{tag}"] = f"{float(v):+.3f}"
    return xmp


def build_iptc(info: dict) -> dict:
    created, offset = capture_datetime(info["exif"])
    iptc = {
        "Iptc.Envelope.CharacterSet": "\x1b%G",      # UTF-8
        "Iptc.Application2.Program": "AI RAW Processor",
        "Iptc.Application2.ProgramVersion": __version__,
        "Iptc.Application2.Keywords": info.get("keywords", [])[:64],
    }
    if info.get("caption"):
        iptc["Iptc.Application2.Caption"] = info["caption"]
        iptc["Iptc.Application2.Headline"] = info["title"]
        iptc["Iptc.Application2.ObjectName"] = info["title"][:64]
    if created:
        iptc["Iptc.Application2.DateCreated"] = created[:10]
        iptc["Iptc.Application2.TimeCreated"] = created[11:19] + (offset or "+00:00")
    if info.get("artist"):
        iptc["Iptc.Application2.Byline"] = [info["artist"]]
    if info.get("copyright"):
        iptc["Iptc.Application2.Copyright"] = info["copyright"]
    loc = info.get("location")
    if loc:
        iptc.update({
            "Iptc.Application2.City": loc["city"][:32],
            "Iptc.Application2.ProvinceState": loc["state"][:32],
            "Iptc.Application2.CountryName": loc["country"][:64],
            "Iptc.Application2.CountryCode": _alpha3(loc["country_code"]),
        })
    return iptc


def _alpha3(cc: str) -> str:
    try:
        import pycountry
        return pycountry.countries.get(alpha_2=cc).alpha_3
    except Exception:
        return cc


def apply(jpeg_bytes: bytes, info: dict) -> bytes:
    img = pyexiv2.ImageData(jpeg_bytes)
    try:
        img.clear_exif()
        img.modify_exif(build_exif(info["exif"], info["width"], info["height"], info.get("caption"),
                                   info.get("artist"), info.get("copyright")))
        img.modify_xmp(build_xmp(info))
        img.modify_iptc(build_iptc(info))
        return img.get_bytes()
    finally:
        img.close()
