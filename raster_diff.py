# raster_diff.py — Pixel-Differencing Engine for CAD Drawing Comparison
# Pipeline: crop live zone → align via homography → absdiff → morphology → classify
#
# Pass 1: Detects ADDED regions only (where V2 is darker than V1).

import cv2
import numpy as np
import fitz
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger("RasterDiff")


# ═══════════════════════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def render_page_gray(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """Render a fitz page to a grayscale numpy array at specified DPI."""
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif pix.n == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    return gray


def render_page_bgr(page: fitz.Page, dpi: int = 300) -> np.ndarray:
    """Render a fitz page to a BGR numpy array at specified DPI."""
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr


# ═══════════════════════════════════════════════════════════════════════════
# LIVE ZONE
# ═══════════════════════════════════════════════════════════════════════════

def compute_live_zone_px(page: fitz.Page, bounds: Dict[str, Any], dpi: int = 300) -> Tuple[int, int, int, int]:
    """Convert the PDF-point live zone to pixel coordinates at the given DPI."""
    scale = dpi / 72.0
    live_zone = bounds.get("live_zone")
    if live_zone:
        return (
            int(live_zone[0] * scale),
            int(live_zone[1] * scale),
            int(live_zone[2] * scale),
            int(live_zone[3] * scale),
        )
    # Fallback: 5% inset from page edges
    w, h = int(page.rect.width * scale), int(page.rect.height * scale)
    return (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))


def crop_to_live_zone(img: np.ndarray, lz: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop an image to the live zone rectangle."""
    x0, y0, x1, y1 = lz
    # Clamp to image bounds
    h, w = img.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return img[y0:y1, x0:x1].copy()


# ═══════════════════════════════════════════════════════════════════════════
# ALIGNMENT (HOMOGRAPHY)
# ═══════════════════════════════════════════════════════════════════════════

def detect_border_corners(gray: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect the four corners of the outermost drawing border.
    Returns 4x2 array of corner points, or None if detection fails.
    """
    # Threshold to get binary (dark lines on white background)
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours — the largest rectangular contour is the border
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort by area descending, pick the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

    # Fallback: use bounding rect corners of largest contour
    x, y, w, h = cv2.boundingRect(contours[0])
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners as: top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],   # top-left (smallest x+y)
        pts[np.argmin(d)],   # top-right (smallest x-y... wait, largest y-x)
        pts[np.argmax(s)],   # bottom-right (largest x+y)
        pts[np.argmax(d)],   # bottom-left (largest y-x)
    ], dtype=np.float32)


def align_images(ref_gray: np.ndarray, mov_gray: np.ndarray) -> np.ndarray:
    """
    Align mov_gray to ref_gray using homography computed from border corners.
    Falls back to identity (no warp) if corner detection fails.
    """
    corners_ref = detect_border_corners(ref_gray)
    corners_mov = detect_border_corners(mov_gray)

    if corners_ref is None or corners_mov is None:
        logger.warning("Border corner detection failed — skipping alignment")
        if mov_gray.shape != ref_gray.shape:
            return cv2.resize(mov_gray, (ref_gray.shape[1], ref_gray.shape[0]),
                              interpolation=cv2.INTER_AREA)
        return mov_gray.copy()

    src = order_corners(corners_mov)
    dst = order_corners(corners_ref)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        logger.warning("Homography computation failed — skipping alignment")
        if mov_gray.shape != ref_gray.shape:
            return cv2.resize(mov_gray, (ref_gray.shape[1], ref_gray.shape[0]),
                              interpolation=cv2.INTER_AREA)
        return mov_gray.copy()

    aligned = cv2.warpPerspective(mov_gray, H, (ref_gray.shape[1], ref_gray.shape[0]),
                                  borderMode=cv2.BORDER_REPLICATE)
    logger.info("Homography alignment applied successfully")
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# PASS 1: RASTER COMPARISON (ADDED ONLY)
# ═══════════════════════════════════════════════════════════════════════════

def raster_compare(
    page1: fitz.Page,
    page2: fitz.Page,
    bounds_v1: Dict[str, Any],
    bounds_v2: Dict[str, Any],
    dpi: int = 300,
    diff_threshold: int = 35,
    intensity_gap: int = 16,
    min_area_pct: float = 0.00005,
) -> Dict[str, Any]:
    """
    Pass 1 raster pixel-differencing pipeline.

    Correct order of operations:
        1. Render V1 and V2 at high DPI
        2. Crop both to live zone
        3. Align V2 to V1 via homography
        4. absdiff -> threshold(35)
        5. Morphological open(5x5), close(15x15)
        6. Connected components, discard < 0.01% of live zone area
        7. Classify: V2 darker by >20 -> ADDED, else skip
        8. Return added_regions[]

    Args:
        diff_threshold: Pixel intensity difference to count as changed (35 for JPG noise)
        intensity_gap:  Minimum mean-intensity difference to classify as ADDED (20)
        min_area_pct:   Minimum blob area as fraction of live zone area (0.0001 = 0.01%)
    """
    scale = dpi / 72.0

    # ── 1. Render ───────────────────────────────────────────────────────
    g1_full = render_page_gray(page1, dpi)
    g2_full = render_page_gray(page2, dpi)

    # Ensure same size before any processing
    if g2_full.shape != g1_full.shape:
        g2_full = cv2.resize(g2_full, (g1_full.shape[1], g1_full.shape[0]),
                             interpolation=cv2.INTER_AREA)

    # ── 2. Crop to drawing area FIRST ───────────────────────────────────
    # NOTE: Stage2's live_zone is designed for vector primitive extraction and
    # uses cluster detection that can produce a partial frame (e.g. only covers
    # top 60% of a large-format page). For pixel diff we need the FULL drawing
    # area. Compute directly from page rect:
    #   - 2% inset on left, right, top (border lines)
    #   - 12% cut from bottom (standard title block zone)
    ph_pt = page1.rect.height
    pw_pt = page1.rect.width

    lz_x0 = int(pw_pt * 0.02 * scale)
    lz_y0 = int(ph_pt * 0.02 * scale)
    lz_x1 = int(pw_pt * 0.98 * scale)
    lz_y1 = int(ph_pt * 0.88 * scale)   # exclude bottom 12% title block
    lz = (lz_x0, lz_y0, lz_x1, lz_y1)

    g1 = crop_to_live_zone(g1_full, lz)
    g2 = crop_to_live_zone(g2_full, lz)

    lz_h, lz_w = g1.shape[:2]
    lz_area_px = lz_w * lz_h
    logger.info(f"Drawing area cropped: {lz_w}x{lz_h} = {lz_area_px}px  "
                f"(page: {pw_pt:.0f}x{ph_pt:.0f}pt, excl. bottom 12%)")

    # ── 3. Align V2 to V1 via homography (only if misaligned) ─────────
    # Vector PDFs rendered at identical DPI by fitz are already pixel-perfect.
    # Homography is only needed for scanned/photographed inputs where physical
    # misalignment exists. Detect by checking if baseline diff is large.
    pre_diff = cv2.absdiff(g1, g2)
    baseline_rmse = float(np.sqrt(np.mean(pre_diff.astype(np.float64) ** 2)))
    logger.info(f"Pre-alignment baseline RMSE: {baseline_rmse:.2f}")

    if baseline_rmse > 15.0:
        # Significant misalignment detected — apply homography
        g2 = align_images(g1, g2)
        logger.info("Homography alignment applied (misalignment detected)")
    else:
        logger.info("Skipping alignment (vector-rendered, already pixel-aligned)")

    # ── 4. absdiff + threshold ──────────────────────────────────────────
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # ── 5. Morphological noise removal ──────────────────────────────────
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # ── 6. Connected components + area filter ───────────────────────────
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = int(lz_area_px * min_area_pct)
    logger.info(f"Min blob area = {min_area}px ({min_area_pct*100:.1f}% of {lz_area_px}px)")

    candidates = []
    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        candidates.append({
            "crop_bbox": (x, y, x + w, y + h),
            "crop_centroid": (cx, cy),
            "px_area": area,
        })

    logger.info(f"Components: {n_labels - 1} raw → {len(candidates)} after {min_area}px area filter")

    # ── 7. Classify: only ADDED (V2 darker than V1) ────────────────────
    added = []

    for c in candidates:
        x0, y0, x1, y1 = c["crop_bbox"]
        region_v1 = g1[y0:y1, x0:x1]
        region_v2 = g2[y0:y1, x0:x1]

        mean_v1 = float(np.mean(region_v1))
        mean_v2 = float(np.mean(region_v2))

        # Pass 1: only report ADDED (V2 has new dark ink)
        if mean_v1 - mean_v2 < intensity_gap:
            # V2 is NOT significantly darker → skip for Pass 1
            continue

        # Convert crop-relative coords back to full-page PDF points
        pt_bbox = [
            (x0 + lz[0]) / scale,
            (y0 + lz[1]) / scale,
            (x1 + lz[0]) / scale,
            (y1 + lz[1]) / scale,
        ]
        pt_centroid = [
            (c["crop_centroid"][0] + lz[0]) / scale,
            (c["crop_centroid"][1] + lz[1]) / scale,
        ]

        added.append({
            "type": "raster-region",
            "status": "ADDED",
            "centroid": pt_centroid,
            "bbox": pt_bbox,
            "area": (pt_bbox[2] - pt_bbox[0]) * (pt_bbox[3] - pt_bbox[1]),
            "px_area": c["px_area"],
            "mean_v1": mean_v1,
            "mean_v2": mean_v2,
            "op_signature": "raster_added",
            "primitive_count": 1,
        })

    logger.info(f"Pass 1 result: {len(added)} ADDED regions")

    return {
        "geometry": {
            "added": added,
            "removed": [],
            "resized": [],
            "unchanged_count": 0,
        },
        "debug": {
            "raw_components": n_labels - 1,
            "after_area_filter": len(candidates),
            "after_classification": len(added),
            "diff_threshold": diff_threshold,
            "intensity_gap": intensity_gap,
            "min_area_px": min_area,
            "min_area_pct": min_area_pct,
            "dpi": dpi,
            "live_zone_px": lz,
            "crop_size": (lz_w, lz_h),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL REPORT
# ═══════════════════════════════════════════════════════════════════════════

def raster_visual_report(
    page1: fitz.Page,
    page2: fitz.Page,
    result: Dict[str, Any],
    drawing_id: str,
    output_dir: str = "visuals",
    dpi: int = 300,
) -> str:
    """
    Generate a 3-panel visual report: V1 | V2 | Analysis (green ADDED boxes).
    Returns the output file path.
    """
    import os

    scale = dpi / 72.0

    bgr1 = render_page_bgr(page1, dpi)
    bgr2 = render_page_bgr(page2, dpi)

    if bgr1.shape != bgr2.shape:
        bgr2 = cv2.resize(bgr2, (bgr1.shape[1], bgr1.shape[0]))

    canvas = bgr2.copy()
    geom = result.get("geometry", {})

    # Draw ADDED boxes (green) — convert PDF-point bbox to pixel coords
    for item in geom.get("added", []):
        bbox = item.get("bbox")
        if bbox and len(bbox) >= 4:
            x0, y0, x1, y1 = [int(v * scale) for v in bbox]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 0), 20)
            cv2.putText(canvas, "ADDED", (x0, max(y0 - 20, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 0), 10)

    # Scale panels to uniform width
    target_w = 1200
    def fit(img):
        h = int(img.shape[0] * target_w / img.shape[1])
        return cv2.resize(img, (target_w, h))

    p1, p2, p3 = fit(bgr1), fit(bgr2), fit(canvas)

    # Headers
    def add_header(img, title):
        hdr = np.full((100, img.shape[1], 3), 35, dtype=np.uint8)
        cv2.putText(hdr, title, (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
        return np.vstack([hdr, img])

    p1 = add_header(p1, "V1 (ORIGINAL)")
    p2 = add_header(p2, "V2 (REVISION)")
    p3 = add_header(p3, f"ANALYSIS: {drawing_id}")

    report = np.hstack([p1, p2, p3])

    # Footer
    n_added = len(geom.get("added", []))
    footer = np.full((80, report.shape[1], 3), 30, dtype=np.uint8)
    txt = f"ID: {drawing_id} | Added: {n_added} | Method: Raster Pixel-Diff (Pass 1)"
    cv2.putText(footer, txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    report = np.vstack([report, footer])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"raster_{drawing_id}.png")
    cv2.imwrite(out_path, report)
    logger.info(f"Raster visual report saved: {out_path}")
    print(f"   [RASTER REPORT] Saved: {out_path}")
    return out_path
