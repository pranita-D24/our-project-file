# balloon_filter.py
# Detects and masks out annotation balloons, callout bubbles,
# revision triangles, and all annotation markers so they are
# never reported as added/removed/modified.

import cv2
import numpy as np
import logging
from config import BALLOON_MIN_RADIUS, BALLOON_MAX_RADIUS, BALLOON_CIRCULARITY

logger = logging.getLogger(__name__)


class BalloonFilter:

    def __init__(self):
        logger.info("BalloonFilter initialized")

    # ─────────────────────────────────────────
    # DETECT CIRCULAR BALLOONS (e.g. item 1, 2)
    # ─────────────────────────────────────────
    def _detect_circle_balloons(self, gray):
        """Hough circles for round annotation bubbles."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=60,
            param2=28,
            minRadius=BALLOON_MIN_RADIUS,
            maxRadius=BALLOON_MAX_RADIUS
        )
        regions = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (cx, cy, r) in circles:
                pad = int(r * 0.35)
                x1 = max(0, cx - r - pad)
                y1 = max(0, cy - r - pad)
                x2 = cx + r + pad
                y2 = cy + r + pad
                regions.append((x1, y1, x2 - x1, y2 - y1))
        return regions

    # ─────────────────────────────────────────
    # DETECT CONTOUR-BASED BALLOONS
    # (handles ellipses, slightly non-circular)
    # ─────────────────────────────────────────
    def _detect_contour_balloons(self, gray):
        """
        Find small highly-circular or elliptical contours
        that look like annotation balloons.
        """
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Slight dilation to close broken circles
        kernel  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300 or area > 12000:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < BALLOON_CIRCULARITY:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            aspect = min(w, h) / (max(w, h) + 1e-6)
            # Must be roughly square (circle/ellipse)
            if aspect < 0.55:
                continue

            pad = 6
            regions.append((
                max(0, x - pad), max(0, y - pad),
                w + 2 * pad, h + 2 * pad))

        return regions

    # ─────────────────────────────────────────
    # DETECT DIMENSION ARROWS / LEADER LINES
    # These small arrowheads near text should
    # also be ignored as annotation
    # ─────────────────────────────────────────
    def _detect_dimension_markers(self, gray):
        """
        Detect small triangular arrowhead markers
        used as dimension line terminators.
        These are tiny triangles at line ends.
        """
        _, binary = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > 800:
                continue

            approx = cv2.approxPolyDP(
                cnt, 0.08 * cv2.arcLength(cnt, True), True)

            # Triangles = arrowheads
            if len(approx) == 3:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 25 and h < 25:
                    regions.append((
                        max(0, x - 3), max(0, y - 3),
                        w + 6, h + 6))

        return regions

    # ─────────────────────────────────────────
    # MASTER: GET ALL BALLOON REGIONS
    # ─────────────────────────────────────────
    def get_balloon_regions(self, gray_image):
        """
        Returns list of (x, y, w, h) bounding boxes
        for all detected balloon/annotation regions.
        """
        r1 = self._detect_circle_balloons(gray_image)
        r2 = self._detect_contour_balloons(gray_image)
        r3 = self._detect_dimension_markers(gray_image)

        all_regions = r1 + r2 + r3
        merged      = self._merge_overlapping(all_regions)

        logger.info(
            f"Balloon regions: {len(merged)} "
            f"(circles:{len(r1)} contours:{len(r2)} "
            f"arrows:{len(r3)})")
        return merged

    # ─────────────────────────────────────────
    # CREATE BALLOON MASK (white = balloon area)
    # ─────────────────────────────────────────
    def create_balloon_mask(self, image_shape, regions):
        """
        Create binary mask where balloon regions = 255.
        Use to exclude these areas from object detection.
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        for (x, y, w, h) in regions:
            cv2.rectangle(mask, (x, y),
                          (x + w, y + h), 255, -1)
        return mask

    # ─────────────────────────────────────────
    # FILTER OBJECTS AGAINST BALLOON MASK
    # ─────────────────────────────────────────
    def filter_objects(self, objects, balloon_mask):
        """
        Remove any detected objects whose centroid
        or bbox overlaps significantly with balloon regions.
        """
        if balloon_mask is None:
            return objects

        filtered = []
        for obj in objects:
            cx, cy = obj["centroid"]
            x, y, w, h = obj["bbox"]

            # Check centroid
            h_m, w_m = balloon_mask.shape
            cx_c = min(cx, w_m - 1)
            cy_c = min(cy, h_m - 1)

            if balloon_mask[cy_c, cx_c] > 0:
                logger.debug(
                    f"Filtered balloon object at {cx},{cy}")
                continue

            # Check bbox overlap
            x2 = min(x + w, w_m)
            y2 = min(y + h, h_m)
            x1 = max(x, 0)
            y1 = max(y, 0)

            if x2 <= x1 or y2 <= y1:
                filtered.append(obj)
                continue

            roi    = balloon_mask[y1:y2, x1:x2]
            overlap = np.sum(roi > 0)
            bbox_area = (x2 - x1) * (y2 - y1)

            # If >35% of bbox is balloon, skip
            if bbox_area > 0 and overlap / bbox_area > 0.35:
                logger.debug(
                    f"Filtered balloon overlap object at {cx},{cy}")
                continue

            filtered.append(obj)

        removed = len(objects) - len(filtered)
        if removed > 0:
            logger.info(
                f"Balloon filter removed {removed} objects")
        return filtered

    # ─────────────────────────────────────────
    # MERGE OVERLAPPING REGIONS
    # ─────────────────────────────────────────
    def _merge_overlapping(self, regions):
        if not regions:
            return []

        rects  = [[x, y, x + w, y + h]
                  for (x, y, w, h) in regions]
        merged = []

        while rects:
            base = rects.pop(0)
            changed = True
            while changed:
                changed    = False
                remaining  = []
                for r in rects:
                    if self._overlaps(base, r):
                        base    = [
                            min(base[0], r[0]),
                            min(base[1], r[1]),
                            max(base[2], r[2]),
                            max(base[3], r[3])]
                        changed = True
                    else:
                        remaining.append(r)
                rects = remaining
            merged.append((
                base[0], base[1],
                base[2] - base[0],
                base[3] - base[1]))
        return merged

    def _overlaps(self, a, b):
        return (a[0] < b[2] and a[2] > b[0] and
                a[1] < b[3] and a[3] > b[1])

    # ─────────────────────────────────────────
    # DEBUG VISUALIZE
    # ─────────────────────────────────────────
    def visualize_balloons(self, image, regions):
        """Draw detected balloon regions for debugging."""
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) \
            if len(image.shape) == 2 \
            else image.copy()
        for (x, y, w, h) in regions:
            cv2.rectangle(vis,
                          (x, y), (x + w, y + h),
                          (255, 0, 255), 2)
            cv2.putText(vis, "BALLOON",
                        (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 0, 255), 1)
        return vis