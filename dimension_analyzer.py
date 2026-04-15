# dimension_analyzer.py — v2.2 FIXED
# Fix Bug 1: _compare_line_lengths proximity reduced 80→50px,
#            added minimum absolute length difference filter,
#            added title-block exclusion zone to skip border lines.

import cv2
import numpy as np
import re
import logging
import os
from config import TESSERACT_CMD, DIM_LINE_MIN_LENGTH, DIM_TEXT_MARGIN

logger = logging.getLogger(__name__)

TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract OCR available ✅")
except Exception:
    logger.warning("Tesseract not available — using CV fallback")


class DimensionAnalyzer:

    DIM_PATTERN = re.compile(
        r'(\d+\.?\d*)'
        r'\s*[\[\(]'
        r'\s*(\d+\'?\s*[-–]?\s*\d*\.?\d+)'
        r'\s*"?\s*[\]\)]'
    )

    STANDALONE_DIM = re.compile(r'\b(\d{2,5}\.?\d*)\b')

    def __init__(self):
        logger.info(
            f"DimensionAnalyzer v2.2 initialized "
            f"(tesseract={TESSERACT_AVAILABLE})")

    # ─────────────────────────────────────────
    # DETECT DRAWING CONTENT BOUNDS
    # Excludes the title block (right ~15% and bottom ~10%)
    # so border lines are never counted as dimension lines.
    # ─────────────────────────────────────────
    def _get_content_roi(self, gray):
        H, W = gray.shape
        # Standard engineering drawing: title block is right 15% and bottom 10%
        x2 = int(W * 0.84)
        y2 = int(H * 0.89)
        return 0, 0, x2, y2

    # ─────────────────────────────────────────
    # DETECT DIMENSION LINES
    # ─────────────────────────────────────────
    def detect_dimension_lines(self, gray):
        # Only look inside the drawing content area, not title block
        x1, y1, x2, y2 = self._get_content_roi(gray)
        roi = gray[y1:y2, x1:x2]

        edges = cv2.Canny(roi, 30, 100)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=50,           # raised from 40 — reduces noise hits
            minLineLength=DIM_LINE_MIN_LENGTH,
            maxLineGap=6)

        dim_lines = []
        if lines is not None:
            for line in lines:
                x1l, y1l, x2l, y2l = line[0]
                length = np.hypot(x2l - x1l, y2l - y1l)
                angle  = abs(np.degrees(
                    np.arctan2(y2l - y1l, x2l - x1l)))
                is_horizontal = angle < 10 or angle > 170   # stricter: was 12/168
                is_vertical   = 80 < angle < 100            # stricter: was 78/102

                if is_horizontal or is_vertical:
                    # Offset back to full image coordinates
                    dim_lines.append({
                        "p1"         : (x1l + x1, y1l + y1),
                        "p2"         : (x2l + x1, y2l + y1),
                        "length"     : round(length, 1),
                        "angle"      : round(angle, 1),
                        "orientation": "H" if is_horizontal else "V"
                    })

        logger.info(f"Dimension lines found: {len(dim_lines)}")
        return dim_lines

    # ─────────────────────────────────────────
    # EXTRACT DIMENSION TEXT (Tesseract)
    # ─────────────────────────────────────────
    def extract_dimension_text_ocr(self, gray):
        if not TESSERACT_AVAILABLE:
            return []

        try:
            import pytesseract

            scale  = 2.5
            scaled = cv2.resize(
                gray,
                (int(gray.shape[1] * scale),
                 int(gray.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC)

            kernel    = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
            sharpened = cv2.filter2D(scaled, -1, kernel)

            clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced  = clahe.apply(sharpened)

            _, thresh = cv2.threshold(
                enhanced, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            data = pytesseract.image_to_data(
                thresh,
                config=(
                    '--psm 6 --oem 3 '
                    '-c tessedit_char_whitelist='
                    "0123456789.[]()\"'-– "
                ),
                output_type=pytesseract.Output.DICT)

            dims = []
            n    = len(data["text"])

            full_text_by_row = {}
            for i in range(n):
                text = data["text"][i].strip()
                if not text or data["conf"][i] < 30:
                    continue
                row_key = data["top"][i] // 15
                if row_key not in full_text_by_row:
                    full_text_by_row[row_key] = []
                full_text_by_row[row_key].append({
                    "text": text,
                    "left": data["left"][i],
                    "top" : data["top"][i],
                    "w"   : data["width"][i],
                    "h"   : data["height"][i],
                    "conf": data["conf"][i],
                })

            for row_key, tokens in full_text_by_row.items():
                tokens.sort(key=lambda t: t["left"])
                row_str = " ".join(t["text"] for t in tokens)

                matches = self.DIM_PATTERN.findall(row_str)
                for m in matches:
                    metric_str   = m[0]
                    imperial_str = m[1]
                    try:
                        metric   = float(metric_str)
                        imperial = self._parse_imperial(imperial_str)
                        if metric > 0:
                            t0 = tokens[0]
                            x  = int(t0["left"] / scale)
                            y  = int(t0["top"]  / scale)
                            w  = int(sum(t["w"] for t in tokens) / scale)
                            h  = int(max(t["h"] for t in tokens) / scale)
                            dims.append({
                                "text"    : f"{metric_str}[{imperial_str}\"]",
                                "metric"  : metric,
                                "imperial": imperial,
                                "bbox"    : (x, y, w, h),
                                "center"  : (x + w // 2, y + h // 2)
                            })
                    except ValueError:
                        continue

            logger.info(f"OCR dimensions found: {len(dims)}")
            return dims

        except Exception as e:
            logger.error(f"OCR error: {e}")
            return []

    def _parse_imperial(self, s):
        s = s.replace("'", "").replace('"', "").strip()
        if '-' in s or '–' in s:
            parts = re.split(r'[-–]', s, maxsplit=1)
            feet  = float(parts[0].strip()) if parts[0].strip() else 0
            inch  = float(parts[1].strip()) if parts[1].strip() else 0
            return round(feet * 12 + inch, 4)
        return float(s)

    # ─────────────────────────────────────────
    # CV FALLBACK
    # ─────────────────────────────────────────
    def detect_text_regions(self, gray):
        mser = cv2.MSER_create(_delta=5, _min_area=20, _max_area=1500)
        _, bboxes = mser.detectRegions(gray)
        text_regions = []
        for bbox in bboxes:
            x, y, w, h = bbox
            aspect = w / (h + 1e-6)
            if 1.2 < aspect < 12 and h < 30:
                text_regions.append((x, y, w, h))
        return self._merge_text_regions(text_regions)

    def _merge_text_regions(self, regions, gap=15):
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: (r[1], r[0]))
        merged  = [list(regions[0])]
        for x, y, w, h in regions[1:]:
            last = merged[-1]
            lx, ly, lw, lh = last
            if abs(y - ly) < 10 and x < lx + lw + gap:
                nx2 = max(lx + lw, x + w)
                ny2 = max(ly + lh, y + h)
                merged[-1] = [min(lx,x), min(ly,y),
                               nx2-min(lx,x), ny2-min(ly,y)]
            else:
                merged.append([x, y, w, h])
        return [tuple(r) for r in merged]

    # ─────────────────────────────────────────
    # COMPARE DIMENSIONS
    # ─────────────────────────────────────────
    def compare_dimensions(self, gray1, gray2):
        result = {
            "lines_v1"           : [],
            "lines_v2"           : [],
            "text_v1"            : [],
            "text_v2"            : [],
            "changed_lines"      : [],
            "added_dims"         : [],
            "removed_dims"       : [],
            "changed_dims"       : [],
            "line_length_changes": [],
            "auto_reading"       : [],
            "summary"            : ""
        }

        try:
            lines1 = self.detect_dimension_lines(gray1)
            lines2 = self.detect_dimension_lines(gray2)
            result["lines_v1"] = lines1
            result["lines_v2"] = lines2

            changed_lengths = self._compare_line_lengths(lines1, lines2)
            result["line_length_changes"] = changed_lengths

            if TESSERACT_AVAILABLE:
                dims1 = self.extract_dimension_text_ocr(gray1)
                dims2 = self.extract_dimension_text_ocr(gray2)
            else:
                dims1 = []
                dims2 = []

            result["text_v1"] = dims1
            result["text_v2"] = dims2

            if dims1 or dims2:
                added, removed, changed = self._compare_dim_texts(dims1, dims2)
                result["added_dims"]   = added
                result["removed_dims"] = removed
                result["changed_dims"] = changed
                result["auto_reading"] = self._build_auto_reading(
                    added, removed, changed, changed_lengths)

            summary_parts = []
            if changed_lengths:
                summary_parts.append(
                    f"{len(changed_lengths)} dimension lines changed")
            if result["changed_dims"]:
                summary_parts.append(
                    f"{len(result['changed_dims'])} dimension values changed")
            if result["added_dims"]:
                summary_parts.append(
                    f"{len(result['added_dims'])} dimensions added")
            if result["removed_dims"]:
                summary_parts.append(
                    f"{len(result['removed_dims'])} dimensions removed")

            result["summary"] = (
                "; ".join(summary_parts)
                if summary_parts
                else "No dimension changes detected")

            logger.info(f"Dimension comparison: {result['summary']}")
            return result

        except Exception as e:
            logger.error(f"Dimension comparison error: {e}")
            result["summary"] = f"Error: {e}"
            return result

    def _build_auto_reading(self, added, removed, changed, changed_lines):
        readings = []
        for ch in changed:
            v1   = ch["v1"]
            v2   = ch["v2"]
            m1   = v1.get("metric")
            m2   = v2.get("metric")
            t1   = v1.get("text", str(m1))
            t2   = v2.get("text", str(m2))
            delta_m = round(m2 - m1, 2) if m1 and m2 else None
            readings.append({
                "type"       : "CHANGED",
                "v1_text"    : t1,
                "v2_text"    : t2,
                "metric_v1"  : m1,
                "metric_v2"  : m2,
                "delta_mm"   : delta_m,
                "description": (
                    f"⚠️  {t1} → {t2}"
                    + (f" (Δ {delta_m:+.1f} mm)" if delta_m else ""))
            })
        for d in added:
            readings.append({
                "type"       : "ADDED",
                "v2_text"    : d.get("text", ""),
                "description": f"🟢 New dimension added: {d.get('text','')}"
            })
        for d in removed:
            readings.append({
                "type"       : "REMOVED",
                "v1_text"    : d.get("text", ""),
                "description": f"🔴 Dimension removed: {d.get('text','')}"
            })
        for ch in changed_lines[:5]:
            readings.append({
                "type"       : "LINE_CHANGED",
                "description": (
                    f"📏 Dimension line ({ch['orientation']}): "
                    f"{ch['v1_length']}px → {ch['v2_length']}px "
                    f"(Δ{ch['change_pct']}%)")
            })
        return readings

    # ─────────────────────────────────────────
    # COMPARE LINE LENGTHS — FIXED
    # Key fixes:
    #   1. Proximity reduced 80→50px: only match lines that are
    #      actually in the same drawing location.
    #   2. Added MIN_ABSOLUTE_DIFF=5px: ignore sub-5px differences
    #      caused by sub-pixel rendering variation, not real changes.
    #   3. Added MIN_PCT_CHANGE=3%: ignore tiny percentage changes
    #      from JPEG compression artifacts.
    # ─────────────────────────────────────────
    def _compare_line_lengths(self, lines1, lines2,
                               tolerance=8.0,
                               proximity=50,        # was 80 — too loose
                               min_abs_diff=5.0,    # NEW: skip sub-pixel noise
                               min_pct_change=3.0): # NEW: skip <3% changes
        changed = []
        used2   = set()

        for l1 in lines1:
            best_match = None
            best_diff  = float('inf')
            best_j     = -1

            for j, l2 in enumerate(lines2):
                if j in used2:
                    continue
                if l1["orientation"] != l2["orientation"]:
                    continue
                c1   = ((l1["p1"][0]+l1["p2"][0])/2,
                        (l1["p1"][1]+l1["p2"][1])/2)
                c2   = ((l2["p1"][0]+l2["p2"][0])/2,
                        (l2["p1"][1]+l2["p2"][1])/2)
                dist = np.hypot(c2[0]-c1[0], c2[1]-c1[1])
                if dist < proximity:
                    diff = abs(l1["length"] - l2["length"])
                    if diff < best_diff:
                        best_diff  = diff
                        best_match = l2
                        best_j     = j

            if best_match and best_diff > tolerance:
                # Skip sub-pixel noise
                if best_diff < min_abs_diff:
                    continue

                pct = round(
                    abs(l1["length"] - best_match["length"])
                    / (l1["length"] + 1e-6) * 100, 1)

                # Skip tiny percentage changes (JPEG/render artifacts)
                if pct < min_pct_change:
                    continue

                changed.append({
                    "v1_length"  : l1["length"],
                    "v2_length"  : best_match["length"],
                    "change_px"  : round(best_diff, 1),
                    "change_pct" : pct,
                    "orientation": l1["orientation"],
                    "v1_line"    : l1,
                    "v2_line"    : best_match
                })
                used2.add(best_j)

        return changed

    # ─────────────────────────────────────────
    # COMPARE DIMENSION TEXT VALUES
    # ─────────────────────────────────────────
    def _compare_dim_texts(self, dims1, dims2, pos_tolerance=60):
        added   = []
        removed = []
        changed = []
        used2   = set()

        for d1 in dims1:
            best_j    = -1
            best_dist = float('inf')
            for j, d2 in enumerate(dims2):
                if j in used2:
                    continue
                cx1, cy1 = d1["center"]
                cx2, cy2 = d2["center"]
                dist = np.hypot(cx2-cx1, cy2-cy1)
                if dist < pos_tolerance and dist < best_dist:
                    best_dist = dist
                    best_j    = j

            if best_j >= 0:
                d2 = dims2[best_j]
                used2.add(best_j)
                m1 = d1.get("metric")
                m2 = d2.get("metric")
                i1 = d1.get("imperial")
                i2 = d2.get("imperial")
                val_changed = False
                if m1 and m2 and abs(m1 - m2) > 0.5:
                    val_changed = True
                if i1 and i2 and abs(i1 - i2) > 0.01:
                    val_changed = True
                if val_changed:
                    changed.append({
                        "v1"             : d1,
                        "v2"             : d2,
                        "metric_change"  : (round(m2-m1, 2) if m1 and m2 else None),
                        "imperial_change": (round(i2-i1, 4) if i1 and i2 else None)
                    })
            else:
                removed.append(d1)

        for j, d2 in enumerate(dims2):
            if j not in used2:
                added.append(d2)

        return added, removed, changed

    # ─────────────────────────────────────────
    # VISUALIZE
    # ─────────────────────────────────────────
    def visualize_dimensions(self, image, dim_result):
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) \
            if len(image.shape) == 2 else image.copy()

        for ch in dim_result.get("line_length_changes", []):
            l = ch.get("v2_line")
            if l:
                cv2.line(vis, l["p1"], l["p2"], (0, 255, 255), 2)
                mid = ((l["p1"][0]+l["p2"][0])//2,
                       (l["p1"][1]+l["p2"][1])//2)
                cv2.putText(vis, f"Δ{ch['change_pct']}%",
                            (mid[0], mid[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 255), 1)

        for ch in dim_result.get("changed_dims", []):
            d = ch["v2"]
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x,y), (x+w,y+h), (255,255,0), 2)
            cv2.putText(vis, f"{ch['v1']['text']}→{ch['v2']['text']}",
                        (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0,255,255), 1)

        return vis