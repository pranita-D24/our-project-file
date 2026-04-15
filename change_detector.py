# change_detector.py — v2.3
# Fix: Added ROI crop before SSIM to exclude title block
# Fix: Verdict correctly weights dimension + object changes
# Fix: Title block no longer inflates pixel difference %

import cv2
import numpy as np
import logging
from skimage.metrics import structural_similarity as ssim
from config import (
    VERY_SIMILAR_THRESHOLD,
    MODERATE_THRESHOLD,
    THRESHOLD_VALUE,
    ROI_X1_PCT, ROI_X2_PCT,
    ROI_Y1_PCT, ROI_Y2_PCT
)

logger = logging.getLogger(__name__)


class ChangeDetector:

    def __init__(self):
        logger.info("ChangeDetector v2.3 initialized")

    def _crop_to_roi(self, image):
        """
        Crop image to drawing area only.
        Excludes title block (right ~20%)
        and notes block (bottom ~22%).
        This prevents revision text / date changes
        in the title block from inflating the
        pixel difference score.
        """
        h, w = image.shape[:2]
        x1   = int(w * ROI_X1_PCT)
        x2   = int(w * ROI_X2_PCT)
        y1   = int(h * ROI_Y1_PCT)
        y2   = int(h * ROI_Y2_PCT)
        return image[y1:y2, x1:x2]

    def detect_changes(self, original, modified_img,
                       match_result, dim_result=None):
        """
        Smart change detection.
        Ignores: moved objects, title block differences
        Detects: added, removed, modified/resized,
                 dimension value changes
        """
        try:
            added         = match_result["added"]
            removed       = match_result["removed"]
            modified_objs = match_result["modified"]
            matched       = match_result["matched"]

            # ── Crop to ROI before SSIM ──
            # KEY FIX: excludes title block from
            # similarity calculation
            orig_roi = self._crop_to_roi(original)
            mod_roi  = self._crop_to_roi(modified_img)

            # ── SSIM on drawing area only ──
            score, diff_map = ssim(
                orig_roi, mod_roi, full=True)
            ssim_similarity = round(score * 100, 2)

            # ── Pixel diff on ROI only ──
            diff_abs       = cv2.absdiff(
                orig_roi, mod_roi)
            _, diff_thresh = cv2.threshold(
                diff_abs, THRESHOLD_VALUE,
                255, cv2.THRESH_BINARY)

            changed_pixels = int(np.sum(diff_thresh > 0))
            total_pixels   = orig_roi.shape[0] * \
                             orig_roi.shape[1]
            changed_ratio  = round(
                changed_pixels / total_pixels * 100, 2)

            # ── Dimension changes count ──
            n_dim_changes = 0
            if dim_result:
                n_dim_changes = (
                    len(dim_result.get(
                        "line_length_changes", [])) +
                    len(dim_result.get(
                        "changed_dims", [])) +
                    len(dim_result.get(
                        "added_dims", [])) +
                    len(dim_result.get(
                        "removed_dims", []))
                )

            total_obj_changes = (
                len(added) +
                len(removed) +
                len(modified_objs)
            )

            # ── SMART VERDICT ──
            # Priority:
            # 1. Real object changes → HIGHLY DIFFERENT
            # 2. Dimension changes   → MODERATELY DIFFERENT
            # 3. Pixel diff          → MODERATELY DIFFERENT
            # 4. Otherwise           → VERY SIMILAR

            if (total_obj_changes >= 5 or
                    changed_ratio >= 15.0):
                verdict    = "HIGHLY DIFFERENT"
                similarity = min(ssim_similarity, 70.0)

            elif (total_obj_changes > 0 or
                  n_dim_changes > 3 or
                  changed_ratio >= 5.0):
                verdict    = "MODERATELY DIFFERENT"
                similarity = min(ssim_similarity, 88.0)

            elif (n_dim_changes > 0 or
                  changed_ratio >= 2.0):
                verdict    = "MODERATELY DIFFERENT"
                similarity = max(
                    ssim_similarity,
                    100 - changed_ratio * 5)
                similarity = min(similarity, 94.0)

            else:
                verdict    = "VERY SIMILAR"
                similarity = ssim_similarity

            similarity = round(similarity, 2)

            auto_summary = self._build_auto_summary(
                added, removed, modified_objs, matched,
                changed_ratio, verdict, ssim_similarity,
                n_dim_changes)

            result = {
                "similarity"        : similarity,
                "ssim_similarity"   : ssim_similarity,
                "difference"        : round(
                    100 - similarity, 2),
                "changed_pixel_pct" : changed_ratio,
                "verdict"           : verdict,
                "added"             : added,
                "removed"           : removed,
                "modified"          : modified_objs,
                "matched"           : matched,
                "added_count"       : len(added),
                "removed_count"     : len(removed),
                "modified_count"    : len(modified_objs),
                "matched_count"     : len(matched),
                "dim_change_count"  : n_dim_changes,
                "diff_map"          : diff_map,
                "diff_abs"          : diff_thresh,
                "auto_summary"      : auto_summary,
            }

            logger.info(
                f"Changes — Verdict:{verdict} "
                f"SSIM:{ssim_similarity}% "
                f"ROI_PixelDiff:{changed_ratio}% "
                f"Added:{len(added)} "
                f"Removed:{len(removed)} "
                f"Modified:{len(modified_objs)} "
                f"DimChanges:{n_dim_changes}")

            return result

        except Exception as e:
            logger.error(f"Change detection error: {e}")
            return None

    def _build_auto_summary(self, added, removed,
                             modified, matched,
                             changed_ratio, verdict,
                             ssim_score,
                             n_dim_changes=0):
        lines = []
        lines.append(f"Verdict: {verdict}")
        lines.append(
            f"Pixel difference: {changed_ratio}% "
            f"of drawing area changed")
        lines.append(f"SSIM score: {ssim_score}%")

        if n_dim_changes:
            lines.append(
                f"Dimension changes detected: "
                f"{n_dim_changes}")
        lines.append("")

        if not added and not removed and not modified:
            if changed_ratio < 2.0 and \
                    n_dim_changes == 0:
                lines.append(
                    "✅ No structural changes detected.")
                lines.append(
                    "   Minor pixel differences may be "
                    "due to rendering or compression.")
            else:
                lines.append(
                    "⚠️  Pixel-level differences found "
                    "but no structural objects changed.")
                if n_dim_changes:
                    lines.append(
                        f"   {n_dim_changes} dimension "
                        f"changes detected.")
                lines.append(
                    "   Check the Dimensions tab for "
                    "specific value changes.")
        else:
            if added:
                lines.append(
                    f"🟢 ADDED ({len(added)} objects):")
                for obj in added[:5]:
                    bbox  = obj.get("bbox", (0,0,0,0))
                    area  = obj.get("area", 0)
                    stype = obj.get(
                        "shape_type", "shape")
                    lines.append(
                        f"   • New {stype} at "
                        f"({bbox[0]},{bbox[1]}) "
                        f"— area {int(area)}px²")
                if len(added) > 5:
                    lines.append(
                        f"   • ...and "
                        f"{len(added)-5} more")

            if removed:
                lines.append(
                    f"🔴 REMOVED ({len(removed)} "
                    f"objects):")
                for obj in removed[:5]:
                    bbox  = obj.get("bbox", (0,0,0,0))
                    area  = obj.get("area", 0)
                    stype = obj.get(
                        "shape_type", "shape")
                    lines.append(
                        f"   • Removed {stype} at "
                        f"({bbox[0]},{bbox[1]}) "
                        f"— area {int(area)}px²")
                if len(removed) > 5:
                    lines.append(
                        f"   • ...and "
                        f"{len(removed)-5} more")

            if modified:
                lines.append(
                    f"🟠 RESIZED/MODIFIED "
                    f"({len(modified)} objects):")
                for pair in modified[:5]:
                    v1   = pair.get("v1_object", {})
                    v2   = pair.get("v2_object", {})
                    pct  = pair.get("area_change", 0)
                    a1   = int(v1.get("area", 0))
                    a2   = int(v2.get("area", 0))
                    bbox = v2.get("bbox", (0,0,0,0))
                    lines.append(
                        f"   • Shape at "
                        f"({bbox[0]},{bbox[1]}): "
                        f"area {a1}px² → {a2}px² "
                        f"({pct:+.1f}%)")
                if len(modified) > 5:
                    lines.append(
                        f"   • ...and "
                        f"{len(modified)-5} more")

        if matched:
            lines.append(
                f"\n⚪ UNCHANGED: {len(matched)} "
                f"objects matched "
                f"(moved objects ignored)")

        return "\n".join(lines)

    def get_change_summary(self, change_result):
        try:
            if not change_result:
                return "No results available"
            if change_result.get("auto_summary"):
                return change_result["auto_summary"]
            lines = [
                f"Similarity : "
                f"{change_result['similarity']}%",
                f"Verdict    : "
                f"{change_result['verdict']}",
                f"Added      : "
                f"{change_result['added_count']} objects",
                f"Removed    : "
                f"{change_result['removed_count']} objects",
                f"Modified   : "
                f"{change_result['modified_count']} objects",
                f"Unchanged  : "
                f"{change_result['matched_count']} objects",
            ]
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return "Error generating summary"


# ══════════════════════════════════════
# TEST
# ══════════════════════════════════════
if __name__ == "__main__":
    import glob, os
    from config       import IMAGE_DIR
    from preprocessor import ImagePreprocessor
    from aligner      import ImageAligner
    from detector     import ObjectDetector
    from matcher      import ObjectMatcher

    print("Testing ChangeDetector v2.3...")
    print("=" * 50)

    prep     = ImagePreprocessor()
    aligner  = ImageAligner()
    detector = ObjectDetector()
    matcher  = ObjectMatcher()
    changer  = ChangeDetector()

    images = glob.glob(
        os.path.join(IMAGE_DIR, "**", "*.jpg"),
        recursive=True)

    if len(images) >= 2:
        orig, mod  = prep.preprocess_pair(
            images[0], images[1])
        aligned    = aligner.align(orig, mod)
        objs1      = detector.detect_objects(orig)
        objs2      = detector.detect_objects(aligned)
        match_res  = matcher.match_objects(objs1, objs2)
        changes    = changer.detect_changes(
            orig, aligned, match_res)

        if changes:
            print(changer.get_change_summary(changes))
            print()
            print(f"ROI Pixel Diff : "
                  f"{changes['changed_pixel_pct']}%")
            print(f"SSIM Score     : "
                  f"{changes['ssim_similarity']}%")
            print(f"Final Similarity: "
                  f"{changes['similarity']}%")
            print("ChangeDetector v2.3 working ✅")
        else:
            print("Detection failed ❌")
    else:
        print("Run pdf_processor.py first!")
    print("=" * 50)