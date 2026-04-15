# ingestion.py — Stage 1 Pipeline Orchestrator
# Single public entry point: ingest(path) -> DrawingProfile
#
# Implements all 7 steps of the ingestion spec:
#   STEP 1 — File routing (.dxf vs PDF/image)
#   STEP 2 — MD5 cache check (cache HIT returns immediately)
#   STEP 3 — Parallel extraction: PyMuPDF raster + pdfplumber text (PDF only)
#   STEP 4 — OpenCV preprocessing: deskew → denoise → CLAHE → Otsu (PDF only)
#   STEP 5 — Layout detection on preprocessed image
#   STEP 6 — Build DrawingProfile
#   STEP 7 — Write cache
#
# The DXF path skips Steps 3-4 entirely (no raster, no CV preprocessing).

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import asdict

logger = logging.getLogger(__name__)


def ingest(path: str):
    """
    Full Stage 1 ingestion pipeline.

    Parameters
    ----------
    path : str
        Absolute or relative path to a PDF, image, or DXF file.

    Returns
    -------
    DrawingProfile
        Fully populated profile. Returned from cache on repeat calls.
    """
    from pdf_reader import (
        DrawingProfile,
        _file_hash,
        CACHE_DIR,
    )

    # ── STEP 1: File routing ─────────────────────────────────────────────────
    ext = pathlib.Path(path).suffix.lower()
    is_dxf = ext == ".dxf"

    # ── STEP 2: MD5 cache check ──────────────────────────────────────────────
    # This is the very first operation after routing — nothing runs on a HIT.
    try:
        fhash      = _file_hash(path)
        cache_file = CACHE_DIR / f"{fhash}.json"

        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            # Deserialise tuple fields stored as JSON arrays
            for k in ("title_block_bbox", "border_bbox", "content_bbox",
                      "gear_data_bbox", "revision_table_bbox"):
                if k in data and isinstance(data[k], list):
                    data[k] = tuple(data[k])
            # Strip private/non-dataclass keys before constructing
            valid_fields = {f.name for f in DrawingProfile.__dataclass_fields__.values()}
            data = {k: v for k, v in data.items() if k in valid_fields}
            logger.info(f"Profile cache HIT: {pathlib.Path(path).name}")
            return DrawingProfile(**data)

    except Exception as e:
        logger.warning(f"Cache read failed for {path}: {e}")
        fhash      = None
        cache_file = None

    logger.info(f"Profile cache MISS: {pathlib.Path(path).name} — profiling...")

    # ── STEPS 3-7: Build profile ─────────────────────────────────────────────
    try:
        if is_dxf:
            # DXF fast-path: Steps 3-4 skipped per spec
            from dxf_reader import build_dxf_profile
            profile = build_dxf_profile(path)
        else:
            # PDF/image path: Steps 3-6 run inside read_and_profile
            from pdf_reader import read_and_profile
            profile = read_and_profile(path)

    except Exception as e:
        logger.error(f"Profile build failed for {path}: {e}")
        profile = DrawingProfile()

    # Optional Vision LLM calibration (non-fatal)
    if not is_dxf:
        try:
            from agent_verifier import calibrate_profile_with_vision
            from pdf_reader import pdf_to_image
            img_bgr = pdf_to_image(path)
            if img_bgr is not None:
                calibrate_profile_with_vision(img_bgr, profile)
        except Exception as e:
            logger.debug(f"Vision calibration skipped: {e}")

    # ── STEP 7: Write cache ───────────────────────────────────────────────────
    if cache_file is not None:
        try:
            # Exclude private/runtime-only attributes not part of the dataclass
            data = asdict(profile)
            cache_file.write_text(
                json.dumps(data, default=str, indent=2),
                encoding="utf-8"
            )
            logger.info(f"Profile cached: {cache_file.name}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    return profile
