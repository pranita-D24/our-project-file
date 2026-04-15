# exclusion.py — Adaptive Exclusion Mask Builder v2.0
# Phase 3 of the Adaptive Intelligence Architecture.
#
# Builds a binary mask of regions to IGNORE during comparison.
# ALL parameters come from DrawingProfile — zero hardcoded geometry.
#
# White pixels (255) = IGNORE
# Black pixels (0)   = ANALYZE

from __future__ import annotations

import logging
from typing import Tuple, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from pdf_reader import DrawingProfile

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 1. TITLE BLOCK
# ────────────────────────────────────────────────────────────
def mask_title_block(gray: np.ndarray,
                      bbox: Tuple[int, int, int, int]) -> np.ndarray:
    H, W = gray.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = 255
        logger.info(f"Title block masked: ({x1},{y1})→({x2},{y2})")
    return mask


# ────────────────────────────────────────────────────────────
# 2. BORDER LINES
# ────────────────────────────────────────────────────────────
def mask_border(gray: np.ndarray,
                border_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Masks everything OUTSIDE the detected border frame.
    Unlike a fixed margin, this uses the actual detected frame lines.
    """
    H, W = gray.shape
    mask = np.full((H, W), 255, dtype=np.uint8)  # start all masked
    x1, y1, x2, y2 = map(int, border_bbox)
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = 0    # unmask the interior
    return mask


# ────────────────────────────────────────────────────────────
# 3. BALLOON ANNOTATIONS
# ────────────────────────────────────────────────────────────
def mask_balloons(gray: np.ndarray,
                  min_r: int = 12,
                  max_r: int = 65) -> np.ndarray:
    """
    Detects and masks balloon annotation circles.
    Radius range comes from DrawingProfile — adaptive to drawing scale.
    """
    H, W = gray.shape
    mask    = np.zeros((H, W), dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # ── Hough circles ──
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=int(min_r * 1.5),
        param1=50, param2=18,
        minRadius=min_r, maxRadius=max_r)

    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(int):
            pad = max(8, r // 3)
            cv2.circle(mask, (cx, cy), r + pad, 255, -1)
        logger.info(f"Hough circles: {len(circles[0])} balloons")

    # ── Contour circularity fallback ──
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated   = cv2.dilate(binary, kernel, iterations=1)
    cnts, _   = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_cnt = 0
    area_min = np.pi * min_r ** 2
    area_max = np.pi * max_r ** 2
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (area_min < area < area_max):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circ   = 4 * np.pi * area / perim ** 2
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = min(w, h) / (max(w, h) + 1e-6)
        if circ > 0.82 and aspect > 0.65:
            pad = max(6, min(w, h) // 4)
            cv2.rectangle(mask,
                          (max(0, x - pad), max(0, y - pad)),
                          (min(W, x + w + pad), min(H, y + h + pad)),
                          255, -1)
            n_cnt += 1

    logger.info(f"Contour balloons: {n_cnt}")
    return mask


# ────────────────────────────────────────────────────────────
# 4. DIMENSION LINES
# ────────────────────────────────────────────────────────────
def mask_dimension_lines(gray: np.ndarray,
                          min_len: int = 30,
                          text_margin: int = 35) -> np.ndarray:
    """
    Masks thin ℕ/℃ dimension lines plus surrounding text area.
    min_len comes from DrawingProfile (scale-dependent).
    text_margin adapts to profile.balloon_radius_min as a proxy for text height.
    """
    H, W = gray.shape
    mask  = np.zeros((H, W), dtype=np.uint8)

    edges = cv2.Canny(gray, 25, 90)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=40, minLineLength=min_len, maxLineGap=6)

    if lines is None:
        return mask

    n_dim = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle  = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        is_h   = angle < 12 or angle > 168
        is_v   = 78 < angle < 102
        if not (is_h or is_v):
            continue

        m = text_margin
        if is_h:
            ymin = max(0, min(y1, y2) - m)
            ymax = min(H, max(y1, y2) + m)
            cv2.rectangle(mask, (min(x1, x2), ymin), (max(x1, x2), ymax), 255, -1)
        else:
            xmin = max(0, min(x1, x2) - m)
            xmax = min(W, max(x1, x2) + m)
            cv2.rectangle(mask, (xmin, min(y1, y2)), (xmax, max(y1, y2)), 255, -1)
        n_dim += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    mask = cv2.dilate(mask, kernel, iterations=2)

    logger.info(f"Dimension lines masked: {n_dim}")
    return mask


# ────────────────────────────────────────────────────────────
# 5. ARBITRARY REGION MASKING
# ────────────────────────────────────────────────────────────
def mask_region(gray: np.ndarray,
                bbox: Tuple[int, int, int, int],
                pad: int = 5) -> np.ndarray:
    H, W = gray.shape
    mask  = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, bbox)
    if x1 < x2 and y1 < y2:
        mask[max(0, y1 - pad):min(H, y2 + pad),
             max(0, x1 - pad):min(W, x2 + pad)] = 255
    return mask


# ────────────────────────────────────────────────────────────
# 6. TEXT REGIONS  (MSER fallback — PaddleOCR not used)
# ────────────────────────────────────────────────────────────
def mask_text_regions_mser(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    mask  = np.zeros((H, W), dtype=np.uint8)

    try:
        mser    = cv2.MSER_create(5, 20, 2000)
        _, bboxes = mser.detectRegions(gray)

        text_regions = []
        for bbox in bboxes:
            x, y, w, h = bbox
            aspect = w / (h + 1e-6)
            if 1.2 < aspect < 15 and h < 50:
                text_regions.append((x, y, w, h))

        # Merge close regions
        text_regions.sort(key=lambda r: (r[1] // 10, r[0]))
        merged: list = []
        used = set()
        for i, (x1, y1, w1, h1) in enumerate(text_regions):
            if i in used:
                continue
            rx2, ry2 = x1 + w1, y1 + h1
            for j, (x2, y2, w2, h2) in enumerate(text_regions[i + 1:], i + 1):
                if j in used:
                    continue
                if abs(y2 - y1) < 12 and x2 < rx2 + 20:
                    rx2 = max(rx2, x2 + w2)
                    ry2 = max(ry2, y2 + h2)
                    used.add(j)
            merged.append((x1, y1, rx2 - x1, ry2 - y1))

        for x, y, w, h in merged:
            pad = 6
            cv2.rectangle(mask,
                          (max(0, x - pad), max(0, y - pad)),
                          (min(W, x + w + pad), min(H, y + h + pad)),
                          255, -1)
        logger.info(f"MSER text regions masked: {len(merged)}")
    except Exception as e:
        logger.warning(f"MSER text masking failed: {e}")

    return mask


# ────────────────────────────────────────────────────────────
# MASTER BUILDER — PROFILE-DRIVEN
# ────────────────────────────────────────────────────────────
def build_exclusion_mask(gray: np.ndarray,
                          profile: Optional["DrawingProfile"] = None,
                          mask_balloons_flag:    bool = True,
                          mask_dimensions_flag:  bool = True,
                          mask_text_flag:        bool = False,
                          mask_border_flag:      bool = True,
                          mask_title_block_flag: bool = True) -> np.ndarray:
    """
    Build complete exclusion mask driven by DrawingProfile.
    If profile is None, falls back to pure content detection (no hardcoded %).

    White (255) = IGNORE  |  Black (0) = ANALYZE
    """
    H, W  = gray.shape
    combined = np.zeros((H, W), dtype=np.uint8)

    # ── Pull params from profile (or use content-detected defaults) ──
    if profile is not None:
        title_bbox   = profile.title_block_bbox
        border_bbox  = profile.border_bbox
        bal_min      = profile.balloon_radius_min
        bal_max      = profile.balloon_radius_max
        dim_min_len  = profile.dim_line_min_length
        dim_margin   = max(20, profile.balloon_radius_min * 2)
        gear_bbox    = profile.gear_data_bbox if profile.has_gear_data_table else (0, 0, 0, 0)
        rev_bbox     = profile.revision_table_bbox if profile.has_revision_table else (0, 0, 0, 0)
        
        # DEBUG: Print masked regions to confirm isometric view is NOT included
        logger.info(f"DEBUG: Exclusion Mask bboxes: Title={title_bbox}, Gear={gear_bbox}, Rev={rev_bbox}, Border={border_bbox}")
    else:
        # Dynamic fallback — run layout detection inline
        from layout_detector import detect_layout
        bgr    = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        layout = detect_layout(bgr)
        title_bbox  = layout["title_block_bbox"]
        border_bbox = layout["border_bbox"]
        bal_min, bal_max = 12, 65
        dim_min_len = 30
        dim_margin  = 35
        gear_bbox   = layout.get("gear_data_bbox",      (0, 0, 0, 0))
        rev_bbox    = layout.get("revision_table_bbox", (0, 0, 0, 0))

    # ── Border / outside-frame ──
    if mask_border_flag and border_bbox != (0, 0, 0, 0):
        m = mask_border(gray, border_bbox)
        combined = cv2.bitwise_or(combined, m)

    # ── Title block ──
    if mask_title_block_flag and title_bbox != (0, 0, 0, 0):
        m = mask_title_block(gray, title_bbox)
        combined = cv2.bitwise_or(combined, m)

    # ── Gear data table ──
    if gear_bbox != (0, 0, 0, 0):
        m = mask_region(gray, gear_bbox)
        combined = cv2.bitwise_or(combined, m)
        logger.info(f"Gear data table masked: {gear_bbox}")

    # ── Revision table ──
    if rev_bbox != (0, 0, 0, 0):
        m = mask_region(gray, rev_bbox)
        combined = cv2.bitwise_or(combined, m)
        logger.info(f"Revision table masked: {rev_bbox}")

    # ── Balloons ──
    if mask_balloons_flag:
        m = mask_balloons(gray, min_r=bal_min, max_r=bal_max)
        combined = cv2.bitwise_or(combined, m)

    # ── Dimension lines ──
    if mask_dimensions_flag:
        m = mask_dimension_lines(gray, min_len=dim_min_len, text_margin=dim_margin)
        combined = cv2.bitwise_or(combined, m)

    # ── Optional text ──
    if mask_text_flag:
        m = mask_text_regions_mser(gray)
        combined = cv2.bitwise_or(combined, m)

    # ── Arbitrary Region Exclusions (Watermarks, stamps) ──
    IGNORE_REGIONS = [
        {"label": "controlled_copy_stamp", "bbox_ratio": (0.25, 0.75, 0.55, 0.95)}
    ]
    for region in IGNORE_REGIONS:
        r = region["bbox_ratio"]
        x1, y1 = int(r[0]*W), int(r[1]*H)
        x2, y2 = int(r[2]*W), int(r[3]*H)
        m = mask_region(gray, (x1, y1, x2, y2))
        combined = cv2.bitwise_or(combined, m)

    # Smooth mask edges slightly
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=1)

    pct = float(np.sum(combined > 0)) / combined.size * 100
    logger.info(f"Exclusion mask covers {pct:.1f}% of image")
    return combined


def build_exclusion_mask_pair(gray1: np.ndarray,
                               gray2: np.ndarray,
                               profile: Optional["DrawingProfile"] = None,
                               p1: Optional["DrawingProfile"] = None,
                               p2: Optional["DrawingProfile"] = None,
                               **kwargs) -> np.ndarray:
    """
    Build combined mask from both V1 and V2.
    Union ensures anything excluded in either version is excluded from both.
    """
    # Only mask title block if it is detected in BOTH versions.
    # If it only exists in one, we want comparison to flag it as ADDED/REMOVED.
    mask_title = True
    if p1 and p2:
        has1 = (p1.title_block_bbox != (0, 0, 0, 0))
        has2 = (p2.title_block_bbox != (0, 0, 0, 0))
        if not (has1 and has2):
            mask_title = False
            logger.info("Title block only in one version — disabling mask to allow change detection.")

    m1 = build_exclusion_mask(gray1, profile=profile, mask_title_block_flag=mask_title, **kwargs)
    m2 = build_exclusion_mask(gray2, profile=profile, mask_title_block_flag=mask_title, **kwargs)
    return cv2.bitwise_or(m1, m2)


# ────────────────────────────────────────────────────────────
# VISUALIZATION (for debugging)
# ────────────────────────────────────────────────────────────
def visualize_exclusion_mask(image_bgr: np.ndarray,
                              mask: np.ndarray,
                              alpha: float = 0.35) -> np.ndarray:
    out     = image_bgr.copy()
    overlay = out.copy()
    overlay[mask > 0] = (0, 0, 200)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 180), 1)
    return out