# layout_detector.py — Content-Based Layout Detection
# Phase 2 of Adaptive Intelligence Architecture.
# Detects title block, content area, border, gear-data table and revision table
# entirely from pixel content — zero hardcoded percentages.

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if len(bgr.shape) == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _binarize(gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def _edge_proximity_score(x1: int, y1: int, x2: int, y2: int,
                           W: int, H: int) -> float:
    """
    Scores how close a rectangle is to ANY image edge.
    0.0 = centred, 1.0 = touching an edge.
    """
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dist_l = cx
    dist_r = W - cx
    dist_t = cy
    dist_b = H - cy
    min_dist = min(dist_l, dist_r, dist_t, dist_b)
    max_half  = max(W, H) / 2
    return max(0.0, 1.0 - min_dist / max_half)


def _aspect_score(w: int, h: int) -> float:
    """
    Title blocks are wide (landscape) or tall (portrait strip) — not square.
    Returns score 0→1 favouring non-square rectangles.
    """
    if w == 0 or h == 0:
        return 0.0
    ratio = max(w, h) / (min(w, h) + 1e-6)
    # Ideal title block ratio: 3:1 – 8:1
    if ratio < 1.2:
        return 0.0   # too square
    return min(1.0, (ratio - 1.2) / 6.0)


# ────────────────────────────────────────────────────────────
# BORDER DETECTION
# ────────────────────────────────────────────────────────────
def detect_border(binary: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Finds the actual drawing frame using Hough lines.
    Returns (x1, y1, x2, y2) bounding the interior of the frame.
    """
    H, W = binary.shape

    lines = cv2.HoughLinesP(
        binary, rho=1, theta=np.pi / 180,
        threshold=200, minLineLength=int(min(W, H) * 0.4), maxLineGap=20)

    if lines is None:
        # Fallback: use projection to find ink boundary
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        thresh = max(h_proj.max(), v_proj.max()) * 0.05
        ys     = np.where(h_proj > thresh)[0]
        xs     = np.where(v_proj > thresh)[0]
        if len(xs) > 0 and len(ys) > 0:
            return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])
        return 0, 0, W, H

    top_y, bot_y, lft_x, rgt_x = H, 0, W, 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        angle  = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        is_h   = angle < 10 or angle > 170
        is_v   = 80 < angle < 100

        if is_h and length > W * 0.3:
            top_y = min(top_y, min(y1, y2))
            bot_y = max(bot_y, max(y1, y2))
        if is_v and length > H * 0.3:
            lft_x = min(lft_x, min(x1, x2))
            rgt_x = max(rgt_x, max(x1, x2))

    if top_y >= bot_y or lft_x >= rgt_x:
        return 0, 0, W, H

    pad = 5
    return (max(0, lft_x + pad), max(0, top_y + pad),
            min(W, rgt_x - pad), min(H, bot_y - pad))


# ────────────────────────────────────────────────────────────
# TITLE BLOCK DETECTION
# ────────────────────────────────────────────────────────────
def detect_title_block(binary: np.ndarray,
                        min_area_frac: float = 0.01,
                        density_thresh: float = 0.15) -> Tuple[int, int, int, int]:
    """
    Content-based title block detection.
    Works for ISO, ANSI, DIN, JIS — any standard.

    Scores every large candidate rectangle by:
      - Text/line pixel density inside it
      - Proximity to any image edge
      - Aspect ratio (title blocks are non-square)

    Returns the highest-scoring bbox (x1,y1,x2,y2) or (0,0,0,0) if none found.
    """
    H, W = binary.shape
    min_area = int(min_area_frac * H * W)

    # Morphological closing to merge nearby text blobs into blocks
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue

        # Text pixel density inside the raw binary
        roi     = binary[y:y + h, x:x + w]
        density = float(np.sum(roi > 0)) / (area + 1e-6)
        if density < density_thresh:
            continue

        prox   = _edge_proximity_score(x, y, x + w, y + h, W, H)
        aspect = _aspect_score(w, h)
        score  = density * 0.5 + prox * 0.35 + aspect * 0.15

        candidates.append((score, (x, y, x + w, y + h)))

    if not candidates:
        return (0, 0, 0, 0)

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


# ────────────────────────────────────────────────────────────
# STRUCTURED TABLE DETECTION  (Gear Data / Revision table)
# ────────────────────────────────────────────────────────────
def detect_structured_tables(binary: np.ndarray,
                               title_block_bbox: Tuple,
                               content_bbox:     Tuple,
                               max_tables: int = 3
                               ) -> List[Tuple[int, int, int, int]]:
    """
    Detects dense rectangular regions that look like data tables
    (parallel horizontal lines crossing vertical dividers).
    Excludes the already-found title block.
    Operates ONLY within the content area.
    """
    cx1, cy1, cx2, cy2 = content_bbox
    tx1, ty1, tx2, ty2 = title_block_bbox

    patch  = binary[cy1:cy2, cx1:cx2].copy()
    H, W   = patch.shape

    # Horizontal line detection
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 8, 1))
    h_lines  = cv2.morphologyEx(patch, cv2.MORPH_OPEN, h_kernel)

    # Vertical line detection
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H // 8))
    v_lines  = cv2.morphologyEx(patch, cv2.MORPH_OPEN, v_kernel)

    # Tables have BOTH horizontal and vertical lines
    table_mask = cv2.bitwise_or(h_lines, v_lines)

    # Dilate to connect a full table region
    expand = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    table_mask = cv2.dilate(table_mask, expand, iterations=1)

    cnts, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tables   = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < int(H * W * 0.005):
            continue
        # Translate back to full image coords
        gx1, gy1 = x + cx1, y + cy1
        gx2, gy2 = gx1 + w, gy1 + h
        # Skip if heavily overlapping with title block
        overlap_x = max(0, min(gx2, tx2) - max(gx1, tx1))
        overlap_y = max(0, min(gy2, ty2) - max(gy1, ty1))
        if overlap_x * overlap_y > area * 0.5:
            continue
        tables.append((gx1, gy1, gx2, gy2))

    tables.sort(key=lambda t: (t[2] - t[0]) * (t[3] - t[1]), reverse=True)
    return tables[:max_tables]


# ────────────────────────────────────────────────────────────
# CONTENT AREA DETECTION
# ────────────────────────────────────────────────────────────
def detect_content_area(binary: np.ndarray,
                          border_bbox:      Tuple[int, int, int, int],
                          title_block_bbox: Tuple[int, int, int, int]
                          ) -> Tuple[int, int, int, int]:
    """
    Finds the main mechanical drawing area.
    = border interior  MINUS  title block region.
    """
    bx1, by1, bx2, by2 = border_bbox
    tx1, ty1, tx2, ty2 = title_block_bbox

    H, W = binary.shape

    # Blank out title block within border region
    mask = np.zeros_like(binary)
    mask[by1:by2, bx1:bx2] = binary[by1:by2, bx1:bx2]
    if tx1 < tx2 and ty1 < ty2:
        mask[ty1:ty2, tx1:tx2] = 0

    # Content boundary via projection
    h_proj = np.sum(mask, axis=1).astype(float)
    v_proj = np.sum(mask, axis=0).astype(float)

    th = max(h_proj.max(), 1) * 0.01
    tv = max(v_proj.max(), 1) * 0.01
    ys = np.where(h_proj > th)[0]
    xs = np.where(v_proj > tv)[0]

    if len(xs) == 0 or len(ys) == 0:
        return bx1, by1, bx2, by2

    pad = 10
    return (max(bx1, int(xs[0]) - pad),
            max(by1, int(ys[0]) - pad),
            min(bx2, int(xs[-1]) + pad),
            min(by2, int(ys[-1]) + pad))


# ────────────────────────────────────────────────────────────
# TABLE CLASSIFICATION  (Gear vs Revision)
# ────────────────────────────────────────────────────────────
def classify_tables(tables: List[Tuple],
                    image_bgr: np.ndarray,
                    full_text: str) -> Dict:
    """
    Given candidate table bboxes, classifies them as gear_data or revision_table.
    Uses OCR on each region for confident classification.
    """
    result = {"gear_data_bbox": (0, 0, 0, 0), "revision_table_bbox": (0, 0, 0, 0)}
    if not tables:
        return result

    try:
        import pytesseract
        for bbox in tables:
            x1, y1, x2, y2 = bbox
            roi   = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            text  = pytesseract.image_to_string(roi, config="--psm 6 --oem 3").upper()
            if any(k in text for k in ["GEAR", "MODULE", "HELIX", "TOOTH"]):
                result["gear_data_bbox"] = bbox
            elif any(k in text for k in ["REV", "REVISION", "ECO", "CHANGE", "DATE"]):
                result["revision_table_bbox"] = bbox
    except Exception as e:
        logger.debug(f"Table classification OCR skipped: {e}")
        # Heuristic fallback: largest table = gear data, second = revision
        if len(tables) >= 1:
            result["gear_data_bbox"] = tables[0]
        if len(tables) >= 2:
            result["revision_table_bbox"] = tables[1]

    return result


# ────────────────────────────────────────────────────────────
# MASTER LAYOUT DETECTOR
# ────────────────────────────────────────────────────────────
def detect_layout(image_bgr: np.ndarray) -> Dict:
    """
    Full layout analysis of a drawing image.
    Returns a dict with:
      border_bbox, title_block_bbox, content_bbox,
      gear_data_bbox, revision_table_bbox
    All coords are (x1, y1, x2, y2) in pixel space.
    """
    H, W = image_bgr.shape[:2]

    gray   = _to_gray(image_bgr)
    binary = _binarize(gray)

    # Step 1: Border frame
    border_bbox = detect_border(binary)
    logger.info(f"Border detected: {border_bbox}")

    # Step 2: Title block
    title_block_bbox = detect_title_block(binary)
    logger.info(f"Title block detected: {title_block_bbox}")

    # Step 3: Content area
    content_bbox = detect_content_area(binary, border_bbox, title_block_bbox)
    logger.info(f"Content area: {content_bbox}")

    # Step 4: Structured tables within content area
    tables = detect_structured_tables(binary, title_block_bbox, content_bbox)
    logger.info(f"Structured tables found: {len(tables)}")

    # Step 5: Classify tables
    table_map = classify_tables(tables, image_bgr, "")

    return {
        "border_bbox":         border_bbox,
        "title_block_bbox":    title_block_bbox,
        "content_bbox":        content_bbox,
        "gear_data_bbox":      table_map.get("gear_data_bbox", (0, 0, 0, 0)),
        "revision_table_bbox": table_map.get("revision_table_bbox", (0, 0, 0, 0)),
    }
