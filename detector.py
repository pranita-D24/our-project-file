# detector.py — v3.0
# Robust object detection for ANY diagram type.
# Key fixes:
# - Adaptive ROI (no more hardcoded 87%/88% crop)
# - Multi-scale detection (catches both tiny and large objects)
# - Smarter balloon/line filter (uses config thresholds)
# - Works on schematics, architectural, CAD, scanned drawings

import cv2
import numpy as np
import logging
from config import (
    MIN_CONTOUR_AREA, MAX_CONTOUR_AREA,
    BALLOON_MIN_RADIUS, BALLOON_MAX_RADIUS,
    BALLOON_CIRCULARITY
)

logger = logging.getLogger(__name__)


class ObjectDetector:

    def __init__(self):
        logger.info("ObjectDetector v3.0 initialized")
        self.orb_patch = cv2.ORB_create(
            400, scaleFactor=1.2, nlevels=6)

    # ─────────────────────────────────────────
    # SHAPE SIGNATURE (48×48 binary silhouette)
    # ─────────────────────────────────────────
    def _shape_signature(self, cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 1 or h <= 1:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        cnt_local = cnt - np.array([[[x, y]]], dtype=cnt.dtype)
        cv2.drawContours(mask, [cnt_local], -1, 255, -1)
        norm = cv2.resize(mask, (48, 48), interpolation=cv2.INTER_AREA)
        return (norm > 0).astype(np.float32).flatten()

    def _patch_signature(self, image, bbox):
        x, y, w, h = bbox
        H, W = image.shape[:2]
        pad_x = max(2, int(0.08*w))
        pad_y = max(2, int(0.08*h))
        x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
        x2 = min(W, x+w+pad_x); y2 = min(H, y+h+pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        patch = cv2.resize(image[y1:y2, x1:x2], (64, 64),
                           interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(patch, 60, 160)
        return (edges > 0).astype(np.float32).flatten()

    def _patch_orb_descriptors(self, image, bbox):
        x, y, w, h = bbox
        H, W = image.shape[:2]
        pad_x = max(2, int(0.08*w))
        pad_y = max(2, int(0.08*h))
        x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
        x2 = min(W, x+w+pad_x); y2 = min(H, y+h+pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        patch    = cv2.resize(image[y1:y2, x1:x2], (96, 96),
                              interpolation=cv2.INTER_AREA)
        patch_u8 = np.clip(patch, 0, 255).astype(np.uint8)
        edges    = cv2.Canny(patch_u8, 50, 150)
        kp, des  = self.orb_patch.detectAndCompute(edges, None)
        if des is None or len(des) == 0:
            return None
        return des[:80] if des.shape[0] > 80 else des

    # ─────────────────────────────────────────
    # FILTERS
    # ─────────────────────────────────────────
    def _is_balloon(self, cnt, area):
        """
        Uses config thresholds (tuned values from config.py).
        More accurate than hardcoded circularity check.
        """
        if area > np.pi * BALLOON_MAX_RADIUS**2 * 1.5:
            return False
        if area < np.pi * BALLOON_MIN_RADIUS**2 * 0.5:
            return False
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter ** 2)
        x, y, w, h  = cv2.boundingRect(cnt)
        aspect      = min(w, h) / (max(w, h) + 1e-6)
        return (circularity > BALLOON_CIRCULARITY and
                aspect > 0.65 and
                area < np.pi * BALLOON_MAX_RADIUS**2)

    def _is_dimension_line(self, cnt, area):
        """Thin elongated contours are dimension lines / leader lines."""
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        return aspect > 15 and area < 2000

    def _is_noise_fragment(self, cnt, area):
        """Very small, irregular blobs are scan/compression noise."""
        if area > MIN_CONTOUR_AREA * 1.5:
            return False
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            return True
        compactness = area / (perimeter + 1e-6)
        return compactness < 1.5

    # ─────────────────────────────────────────
    # ADAPTIVE ROI DETECTION
    # ─────────────────────────────────────────
    def _detect_drawing_roi(self, gray):
        """
        Automatically find the drawing content area.
        Handles:
        - Title blocks (right side or bottom)
        - Border lines
        - Blank scan margins

        Returns (x1, y1, x2, y2) bounding box of content area.
        Falls back to 95% of image if detection fails.
        """
        H, W = gray.shape

        # Find content via horizontal/vertical projection
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection (sum across columns)
        h_proj = np.sum(binary, axis=1).astype(np.float32)
        v_proj = np.sum(binary, axis=0).astype(np.float32)

        # Smooth projections
        h_proj = cv2.GaussianBlur(
            h_proj.reshape(-1, 1), (1, 21), 0).flatten()
        v_proj = cv2.GaussianBlur(
            v_proj.reshape(1, -1), (21, 1), 0).flatten()

        threshold_h = max(h_proj) * 0.03
        threshold_v = max(v_proj) * 0.03

        h_active = np.where(h_proj > threshold_h)[0]
        v_active = np.where(v_proj > threshold_v)[0]

        if len(h_active) == 0 or len(v_active) == 0:
            # Fallback: use 95% of image, exclude edges
            margin = int(min(H, W) * 0.02)
            return margin, margin, W - margin, H - margin

        y1 = max(0,   int(h_active[0])  - 5)
        y2 = min(H,   int(h_active[-1]) + 5)
        x1 = max(0,   int(v_active[0])  - 5)
        x2 = min(W,   int(v_active[-1]) + 5)

        # Detect title block: large dense region on right side
        # Typical title block = rightmost 12-15% of width
        right_strip = binary[:, int(W * 0.85):]
        right_density = np.mean(right_strip)
        if right_density > 10:  # title block present
            x2 = min(x2, int(W * 0.85))
            logger.info("Title block detected — excluding from ROI")

        # Detect title block on bottom
        bottom_strip = binary[int(H * 0.90):, :]
        bottom_density = np.mean(bottom_strip)
        if bottom_density > 10:
            y2 = min(y2, int(H * 0.90))
            logger.info("Bottom block detected — excluding from ROI")

        logger.info(f"Drawing ROI: ({x1},{y1}) → ({x2},{y2}) "
                    f"of ({W},{H})")
        return x1, y1, x2, y2

    # ─────────────────────────────────────────
    # BUILD OBJECT RECORD
    # ─────────────────────────────────────────
    def _build_object(self, cnt_full, full_image, obj_id):
        """Extract all descriptors for one contour."""
        x, y, w, h  = cv2.boundingRect(cnt_full)
        area        = float(cv2.contourArea(cnt_full))
        perimeter   = float(cv2.arcLength(cnt_full, True))
        circularity = (4 * np.pi * area /
                       (perimeter ** 2 + 1e-6))

        M  = cv2.moments(cnt_full)
        cx = int(M["m10"]/M["m00"]) if M["m00"] != 0 else x+w//2
        cy = int(M["m01"]/M["m00"]) if M["m00"] != 0 else y+h//2

        hu      = cv2.HuMoments(cv2.moments(cnt_full)).flatten()
        approx  = cv2.approxPolyDP(cnt_full, 0.04*perimeter, True)
        sides   = len(approx)

        if sides == 3:
            shape = "triangle"
        elif sides == 4:
            ar = w / (h + 1e-6)
            shape = "square" if 0.9 < ar < 1.1 else "rectangle"
        elif circularity > 0.70:
            shape = "circle"
        else:
            shape = "complex"

        return {
            "object_id"      : obj_id,
            "contour"        : cnt_full,
            "area"           : area,
            "bbox"           : (x, y, w, h),
            "centroid"       : (cx, cy),
            "circularity"    : float(circularity),
            "hu_moments"     : hu,
            "shape_signature": self._shape_signature(cnt_full),
            "patch_signature": self._patch_signature(
                full_image, (x, y, w, h)),
            "patch_orb"      : self._patch_orb_descriptors(
                full_image, (x, y, w, h)),
            "shape_type"     : shape,
            "perimeter"      : float(perimeter),
            "is_balloon"     : False
        }

    # ─────────────────────────────────────────
    # MULTI-SCALE DETECTION
    # ─────────────────────────────────────────
    def _detect_at_scale(self, roi, roi_offset,
                          full_image, min_area, max_area,
                          kernel_size=(2,2)):
        """
        Detect objects at one scale level.
        Returns list of (cnt_full, area) tuples.
        """
        ROI_X, ROI_Y = roi_offset

        _, binary = cv2.threshold(
            roi, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Adaptive threshold as fallback if Otsu gives poor result
        dark_pct = np.sum(binary > 0) / binary.size
        if dark_pct > 0.6 or dark_pct < 0.01:
            binary = cv2.adaptiveThreshold(
                roi, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=15, C=8)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, kernel_size)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue

            # Offset to full image coordinates
            cnt_full = cnt + np.array(
                [[[ROI_X, ROI_Y]]], dtype=cnt.dtype)

            if self._is_balloon(cnt, area):
                continue
            if self._is_dimension_line(cnt, area):
                continue
            if self._is_noise_fragment(cnt, area):
                continue

            results.append((cnt_full, area))

        return results

    # ─────────────────────────────────────────
    # MERGE DUPLICATE DETECTIONS
    # ─────────────────────────────────────────
    def _merge_detections(self, all_cnts, iou_threshold=0.5):
        """
        Remove duplicate contours from multi-scale detection
        by merging overlapping bounding boxes.
        """
        if not all_cnts:
            return []

        bboxes = [cv2.boundingRect(c) for c, _ in all_cnts]
        keep   = []
        used   = set()

        for i, (cnt_i, area_i) in enumerate(all_cnts):
            if i in used:
                continue
            xi, yi, wi, hi = bboxes[i]
            merged = False
            for j in range(i+1, len(all_cnts)):
                if j in used:
                    continue
                xj, yj, wj, hj = bboxes[j]
                # IoU
                ix1 = max(xi, xj); iy1 = max(yi, yj)
                ix2 = min(xi+wi, xj+wj)
                iy2 = min(yi+hi, yj+hj)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1) * (iy2-iy1)
                    union = wi*hi + wj*hj - inter
                    iou   = inter / (union + 1e-6)
                    if iou > iou_threshold:
                        # Keep the larger one
                        if area_i >= (all_cnts[j][1]):
                            used.add(j)
                        else:
                            used.add(i)
                            merged = True
                            break
            if not merged and i not in used:
                keep.append((cnt_i, area_i))

        return keep

    # ─────────────────────────────────────────
    # MAIN DETECT
    # ─────────────────────────────────────────
    def detect_objects(self, image, balloon_mask=None):
        """
        Detect all meaningful objects in any diagram.

        Strategy:
        1. Auto-detect drawing ROI (adaptive, no hardcoded %)
        2. Multi-scale detection (normal + fine-detail pass)
        3. Merge duplicates
        4. Extract full descriptors
        """
        try:
            full_image = image.copy()
            H, W = image.shape[:2]

            # ── Auto ROI ──
            x1, y1, x2, y2 = self._detect_drawing_roi(image)
            roi    = image[y1:y2, x1:x2]
            offset = (x1, y1)

            # ── Pass 1: Normal scale ──
            normal_detections = self._detect_at_scale(
                roi, offset, full_image,
                min_area=MIN_CONTOUR_AREA,
                max_area=MAX_CONTOUR_AREA,
                kernel_size=(2, 2))

            # ── Pass 2: Fine detail (smaller min area) ──
            # Catches small components like bolt holes,
            # connector pins, small annotation boxes
            fine_detections = self._detect_at_scale(
                roi, offset, full_image,
                min_area=max(100, MIN_CONTOUR_AREA // 4),
                max_area=MIN_CONTOUR_AREA * 3,
                kernel_size=(1, 1))

            all_detections = normal_detections + fine_detections

            # ── Merge overlapping boxes ──
            merged = self._merge_detections(all_detections)

            # ── Build objects ──
            objects = []
            for cnt_full, area in merged:
                x, y, w, h = cv2.boundingRect(cnt_full)

                # Skip if center is in balloon mask
                if balloon_mask is not None:
                    cx_b = min(x + w//2, balloon_mask.shape[1]-1)
                    cy_b = min(y + h//2, balloon_mask.shape[0]-1)
                    if balloon_mask[cy_b, cx_b] > 0:
                        continue

                obj = self._build_object(
                    cnt_full, full_image, len(objects))
                objects.append(obj)

            logger.info(
                f"Detected {len(objects)} objects "
                f"(from {len(all_detections)} raw detections "
                f"→ {len(merged)} after merge)")
            return objects

        except Exception as e:
            logger.error(f"Detection error: {e}")
            import traceback; traceback.print_exc()
            return []

    def draw_objects(self, image, objects, color=(0,255,0)):
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) \
            if len(image.shape) == 2 else image.copy()
        for obj in objects:
            x, y, w, h = obj["bbox"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
            cv2.putText(vis, f"{obj['object_id']}",
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1)
        return vis