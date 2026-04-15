# dxf_reader.py — Lightweight DXF Drawing Profile Builder
# Stage 1, DXF fast-path. Skips Steps 3-4 (no raster rendering, no CV preprocessing).
# Reads ezdxf metadata only → populates DrawingProfile → returns for caching.

from __future__ import annotations

import logging
import math
import pathlib
from typing import Optional

logger = logging.getLogger(__name__)

# Keywords for drawing-standard detection from text entities
_STANDARD_KEYWORDS = {
    "ISO":  ["ISO 128", "ISO128", "ISO 2768", "ISO", "BS EN ISO"],
    "DIN":  ["DIN 128", "DIN", "DIN ISO"],
    "ANSI": ["ANSI", "ASME", "ASME Y14", "ANSI Y14"],
    "JIS":  ["JIS", "JIS B"],
    "BS":   ["BS", "BS 308", "BS8888"],
    "AS":   ["AS 1100", "AS1100"],
}


def _detect_standard_from_text(text: str) -> str:
    upper = text.upper()
    for std, keywords in _STANDARD_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in upper:
                return std
    return "UNKNOWN"


def _is_valid_coord(v) -> bool:
    """Returns False if coordinate is infinite, NaN, or absurdly large."""
    try:
        f = float(v)
        return math.isfinite(f) and abs(f) < 1e9
    except (TypeError, ValueError):
        return False


def _safe_bbox(extmin, extmax) -> tuple:
    """
    Convert ezdxf extents to (x1, y1, x2, y2) int bbox.
    Falls back to (0, 0, 1000, 1000) if any coordinate is invalid.
    """
    try:
        x1, y1 = extmin[0], extmin[1]
        x2, y2 = extmax[0], extmax[1]
        if all(_is_valid_coord(v) for v in (x1, y1, x2, y2)):
            # Normalise so x1 < x2 and y1 < y2
            lx, rx = (int(min(x1, x2)), int(max(x1, x2)))
            ly, ry = (int(min(y1, y2)), int(max(y1, y2)))
            if rx > lx and ry > ly:
                return lx, ly, rx, ry
    except Exception:
        pass

    logger.warning(
        "DXF extents are missing or invalid (infinite / NaN / out of range). "
        "Falling back to content_bbox = (0, 0, 1000, 1000)."
    )
    return 0, 0, 1000, 1000


def build_dxf_profile(path: str):
    """
    Build a DrawingProfile from a DXF file without rasterizing.
    Steps 3 and 4 of the ingestion pipeline are intentionally skipped.

    Returns a DrawingProfile suitable for caching.
    """
    from pdf_reader import DrawingProfile  # avoid circular at module level

    profile = DrawingProfile()

    try:
        import ezdxf

        doc = ezdxf.readfile(path)
        hdr = doc.header

        # ── Units (STEP 6 field: units) ──────────────────────────
        # $MEASUREMENT: 0 = imperial (inches), 1 = metric (mm)
        measurement = hdr.get("$MEASUREMENT", 1)
        profile.units = "mm" if measurement == 1 else "inch"

        # ── Drawing standard from text entities ─────────────────
        text_blob = []
        msp = doc.modelspace()
        for ent in msp:
            if ent.dxftype() in ("TEXT", "MTEXT"):
                try:
                    text_blob.append(ent.dxf.text if ent.dxftype() == "TEXT"
                                     else ent.text)
                except Exception:
                    pass
        full_text = " ".join(text_blob)
        profile.drawing_standard = _detect_standard_from_text(full_text)

        # ── Extents → content_bbox ───────────────────────────────
        extmin = hdr.get("$EXTMIN", (0, 0, 0))
        extmax = hdr.get("$EXTMAX", (0, 0, 0))
        profile.content_bbox = _safe_bbox(extmin, extmax)

        # ── Border bbox = same as content for DXF ───────────────
        profile.border_bbox = profile.content_bbox

        # ── Scale from $DIMSCALE ────────────────────────────────
        dimscale = hdr.get("$DIMSCALE", 1.0)
        if dimscale and _is_valid_coord(dimscale) and dimscale > 0:
            profile.scale_ratio = float(dimscale)
            profile.scale = f"1:{int(1/dimscale)}" if dimscale < 1 else f"{int(dimscale)}:1"

        # ── Drawing number from filename (best effort) ──────────
        profile.drawing_number = pathlib.Path(path).stem

        logger.info(
            f"DXF profile built: units={profile.units} std={profile.drawing_standard} "
            f"content_bbox={profile.content_bbox} scale={profile.scale}"
        )

    except Exception as e:
        logger.error(f"build_dxf_profile failed for {path}: {e}")
        # Return default profile with safe bbox
        profile.content_bbox = (0, 0, 1000, 1000)
        profile.border_bbox  = (0, 0, 1000, 1000)

    return profile
