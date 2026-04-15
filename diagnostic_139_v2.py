"""
DIAGNOSTIC — PRV73B124139 raster_diff crop boundary failure
Reports all 6 checks as JSON. Does NOT modify any source code.
"""
import sys, os, json, fitz, numpy as np, cv2, logging
sys.path.insert(0, os.path.abspath('.'))

from stage2_vector import Stage2Engine
from raster_diff import render_page_gray, crop_to_live_zone

def run_diagnostic():
    drawing_id = "PRV73B124139"
    v1_path = r"Drawings\PRV73B124139.PDF"
    v2_path = r"Drawings\PRV73B124139 - Copy.PDF"

    doc1 = fitz.open(v1_path)
    doc2 = fitz.open(v2_path)
    page1 = doc1[0]
    page2 = doc2[0]

    # ── CHECK 1: Page dimensions ──────────────────────────────────────
    pw, ph = page1.rect.width, page1.rect.height
    print(f"[CHECK 1] Page rect: {pw:.2f} x {ph:.2f} pt")

    # ── CHECK 2: Stage2 outer frame ───────────────────────────────────
    s2 = Stage2Engine()
    bounds_v1 = s2.detect_boundaries(page1)
    bounds_v2 = s2.detect_boundaries(page2)
    
    of = bounds_v2["outer_frame"]
    of_rect = {"x0": round(of.x0, 2), "y0": round(of.y0, 2),
               "x1": round(of.x1, 2), "y1": round(of.y1, 2)}
    of_flag = "PASS" if of.y1 > 1000 else f"FAIL - partial frame detected (y1={of.y1:.1f} < 1000)"
    print(f"[CHECK 2] Outer frame: {of_rect}")
    print(f"          Flag: {of_flag}")

    # ── CHECK 3: Live zone ────────────────────────────────────────────
    lz = bounds_v2.get("live_zone")
    lz_rect = {"x0": round(lz[0], 2), "y0": round(lz[1], 2),
               "x1": round(lz[2], 2), "y1": round(lz[3], 2)}
    lz_flag = "PASS" if lz[3] > 800 else f"FAIL - dimension will be cropped out (y1={lz[3]:.1f} < 800)"
    print(f"[CHECK 3] Live zone: {lz_rect}")
    print(f"          Flag: {lz_flag}")

    # ── CHECK 4: 175.65 dimension text location ───────────────────────
    dim_y = None
    dim_text_found = None
    t = page2.get_text("dict")
    for b in t.get("blocks", []):
        if "lines" not in b:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                if "175" in s["text"]:
                    dim_y = s["bbox"][1]
                    dim_text_found = s["text"].strip()
                    print(f"[CHECK 4] Found '{dim_text_found}' at y={dim_y:.2f}  bbox={s['bbox']}")

    if dim_y is None:
        dim_text_found = "NOT FOUND"
        dim_y = -1
        dim_175_flag = "FAIL - text not found on page"
    elif dim_y > lz[3]:
        dim_175_flag = f"FAIL - element outside crop boundary (y={dim_y:.1f} > lz_y1={lz[3]:.1f})"
    else:
        dim_175_flag = "PASS"
    print(f"          Flag: {dim_175_flag}")

    # ── CHECK 5: Raster crop source ───────────────────────────────────
    # Read what raster_diff.py actually does
    with open('raster_diff.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if "ph_pt = page1.rect.height" in content and "lz_y1 = int(ph_pt * 0.88 * scale)" in content:
        raster_crop_source = "page_rect (manual 2%/88% inset)"
        raster_crop_flag = "PASS - uses page_rect directly, not Stage2 live_zone"
    elif "compute_live_zone_px" in content and "bounds" in content:
        raster_crop_source = "live_zone (from Stage2 detect_boundaries)"
        raster_crop_flag = "FAIL - should use page_rect for raster diff"
    else:
        raster_crop_source = "unknown"
        raster_crop_flag = "UNKNOWN - could not determine source"
    print(f"[CHECK 5] Raster crop source: {raster_crop_source}")
    print(f"          Flag: {raster_crop_flag}")

    # ── CHECK 6: Diff output ─────────────────────────────────────────
    dpi = 300
    diff_threshold = 35
    intensity_gap = 20
    min_area_pct = 0.00005
    scale = dpi / 72.0

    g1_full = render_page_gray(page1, dpi)
    g2_full = render_page_gray(page2, dpi)
    if g2_full.shape != g1_full.shape:
        g2_full = cv2.resize(g2_full, (g1_full.shape[1], g1_full.shape[0]),
                             interpolation=cv2.INTER_AREA)

    # Replicate the exact crop logic from raster_diff.py
    ph_pt = page1.rect.height
    pw_pt = page1.rect.width
    clz_x0 = int(pw_pt * 0.02 * scale)
    clz_y0 = int(ph_pt * 0.02 * scale)
    clz_x1 = int(pw_pt * 0.98 * scale)
    clz_y1 = int(ph_pt * 0.88 * scale)
    clz = (clz_x0, clz_y0, clz_x1, clz_y1)

    print(f"\n[CHECK 6] Raster crop zone (px): {clz}")
    print(f"          Full render size: {g1_full.shape[1]}x{g1_full.shape[0]}")
    print(f"          Crop bottom (pt): {ph_pt * 0.88:.1f} of {ph_pt:.1f}")

    # Also check: where does the 175 dimension land in pixel space?
    if dim_y and dim_y > 0:
        dim_y_px = int(dim_y * scale)
        print(f"          175 dim y in px: {dim_y_px}  |  crop y1 px: {clz_y1}")
        if dim_y_px > clz_y1:
            print(f"          >>> 175 dim IS BEING CROPPED OUT ({dim_y_px} > {clz_y1})")
        else:
            print(f"          >>> 175 dim is within crop zone")

    g1 = g1_full[clz_y0:clz_y1, clz_x0:clz_x1]
    g2 = g2_full[clz_y0:clz_y1, clz_x0:clz_x1]

    # absdiff + threshold + morphology
    diff = cv2.absdiff(g1, g2)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lz_area_px = (clz_x1 - clz_x0) * (clz_y1 - clz_y0)
    min_area = int(lz_area_px * min_area_pct)

    blob_list = []
    added_boxes = 0
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        region_v1 = g1[y:y+h, x:x+w]
        region_v2 = g2[y:y+h, x:x+w]
        m1 = float(np.mean(region_v1))
        m2 = float(np.mean(region_v2))

        status = "SKIPPED"
        reason = f"gap={m1 - m2:.2f} < {intensity_gap}"
        if m1 - m2 >= intensity_gap:
            status = "ADDED"
            reason = f"V2 darker by {m1 - m2:.2f} >= {intensity_gap}"
            added_boxes += 1

        blob_list.append({
            "bbox_px": [int(x), int(y), int(x+w), int(y+h)],
            "area": int(area),
            "status": status,
            "reason": reason,
            "mean_v1": round(m1, 2),
            "mean_v2": round(m2, 2),
            "gap": round(m1 - m2, 2)
        })

    print(f"\n          Raw components: {n_labels - 1}")
    print(f"          After area filter: {len(blob_list)} (min_area={min_area})")
    print(f"          ADDED boxes: {added_boxes}")
    for idx, blob in enumerate(blob_list):
        print(f"          Blob {idx+1}: {blob}")

    # ── BUILD VERDICT ─────────────────────────────────────────────────
    if dim_y and dim_y > 0:
        dim_in_crop_pt = ph_pt * 0.88
        if dim_y > dim_in_crop_pt:
            verdict = (f"ROOT CAUSE: 175 dimension at y={dim_y:.1f}pt is BELOW the raster crop "
                       f"boundary ({dim_in_crop_pt:.1f}pt = 88% of page height {ph_pt:.1f}pt). "
                       f"The 12% bottom exclusion intended for title block also clips the 175 dimension.")
        elif of.y1 < 1000:
            verdict = (f"ROOT CAUSE: Stage2 outer_frame.y1={of.y1:.1f} < 1000pt indicates partial "
                       f"frame detection, BUT raster_diff uses page_rect directly so this should not "
                       f"affect raster crop. Check if 175 dim is being missed in classification.")
        else:
            verdict = "No obvious crop boundary failure detected — dim is within crop zone."
    else:
        verdict = "175 dimension text NOT FOUND on V2 page — check PDF content."

    # ── JSON REPORT ───────────────────────────────────────────────────
    report = {
        "page_rect": {"w": round(pw, 2), "h": round(ph, 2)},
        "outer_frame": of_rect,
        "outer_frame_flag": of_flag,
        "live_zone": lz_rect,
        "live_zone_flag": lz_flag,
        "dim_175_y_coord": round(dim_y, 2) if dim_y else -1,
        "dim_175_text": dim_text_found,
        "dim_175_flag": dim_175_flag,
        "raster_crop_source": raster_crop_source,
        "raster_crop_flag": raster_crop_flag,
        "raster_crop_zone_px": list(clz),
        "raster_crop_bottom_pt": round(ph_pt * 0.88, 2),
        "blobs_found": len(blob_list),
        "blob_list": blob_list,
        "added_boxes": added_boxes,
        "verdict": verdict
    }

    print("\n" + "=" * 70)
    print("DIAGNOSTIC REPORT (JSON):")
    print("=" * 70)
    print(json.dumps(report, indent=2))

    doc1.close()
    doc2.close()

if __name__ == "__main__":
    run_diagnostic()
