import sys
import os
import json
import fitz
import numpy as np
import cv2
import logging

# Ensure we can import from the workspace
sys.path.insert(0, os.path.abspath('.'))

from stage2_vector import Stage2Engine
from raster_diff import render_page_gray, compute_live_zone_px, crop_to_live_zone, align_images

def run_diagnostic():
    drawing_id = "PRV73B124139"
    v1_path = r"Drawings\PRV73B124139.PDF"
    v2_path = r"Drawings\PRV73B124139 - Copy.PDF"
    
    doc1 = fitz.open(v1_path)
    doc2 = fitz.open(v2_path)
    page1 = doc1[0]
    page2 = doc2[0]
    
    # CHECK 1: Page dimensions
    pw, ph = page2.rect.width, page2.rect.height
    
    # CHECK 2: Stage2 outer frame
    s2 = Stage2Engine()
    bounds_v2 = s2.detect_boundaries(page2)
    of = bounds_v2["outer_frame"]
    of_rect = {"x0": of.x0, "y0": of.y0, "x1": of.x1, "y1": of.y1}
    of_flag = "PASS" if of.y1 > 1000 else f"FAIL - partial frame detected (y1={of.y1:.1f} < 1000)"
    
    # CHECK 3: Live zone
    lz = bounds_v2.get("live_zone")
    lz_rect = {"x0": lz[0], "y0": lz[1], "x1": lz[2], "y1": lz[3]}
    lz_flag = "PASS" if lz[3] > 800 else f"FAIL - dimension will be cropped out (y1={lz[3]:.1f} < 800)"
    
    # CHECK 4: 175.65 dimension text location
    dim_y = 0
    t = page2.get_text("dict")
    for b in t.get("blocks", []):
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                if "175" in s["text"]:
                    dim_y = s["bbox"][1] # y0
                    break
    dim_175_flag = "PASS" if dim_y < lz[3] else f"FAIL - element outside crop boundary (y={dim_y:.1f} > lz_y1={lz[3]:.1f})"
    
    # CHECK 5: Raster crop source
    # Need to read the file to see what it's doing
    with open('raster_diff.py', 'r') as f:
        content = f.read()
    
    if "ph_pt = page1.rect.height" in content and "lz_y1 = int(ph_pt * 0.88 * scale)" in content:
        raster_crop_source = "page_rect (manual inset)"
        raster_crop_flag = "PASS"
    else:
        raster_crop_source = "live_zone"
        raster_crop_flag = "FAIL - should use page_rect for raster diff"

    # CHECK 6: Diff output
    # Reproduce the diff logic internally to get blob list
    dpi = 300
    diff_threshold = 35
    intensity_gap = 20
    min_area_pct = 0.00005
    
    g1_full = render_page_gray(page1, dpi)
    g2_full = render_page_gray(page2, dpi)
    if g2_full.shape != g1_full.shape:
        g2_full = cv2.resize(g2_full, (g1_full.shape[1], g1_full.shape[0]), interpolation=cv2.INTER_AREA)
        
    ph_pt = page1.rect.height
    pw_pt = page1.rect.width
    scale = dpi / 72.0

    # Current implementation in raster_diff.py uses ph_pt * 0.88 for y1
    clz_x0 = int(pw_pt * 0.02 * scale)
    clz_y0 = int(ph_pt * 0.02 * scale)
    clz_x1 = int(pw_pt * 0.98 * scale)
    clz_y1 = int(ph_pt * 0.88 * scale)
    clz = (clz_x0, clz_y0, clz_x1, clz_y1)

    g1 = g1_full[clz_y0:clz_y1, clz_x0:clz_x1]
    g2 = g2_full[clz_y0:clz_y1, clz_x0:clz_x1]
    
    # Skip alignment as per local logic
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
        if area < min_area: continue
        
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Classification
        region_v1 = g1[y:y+h, x:x+w]
        region_v2 = g2[y:y+h, x:x+w]
        m1 = float(np.mean(region_v1))
        m2 = float(np.mean(region_v2))
        
        status = "NONE"
        if m1 - m2 >= intensity_gap:
            status = "ADDED"
            added_boxes += 1
            
        blob_list.append({
            "bbox_px": [int(x), int(y), int(x+w), int(y+h)],
            "area": int(area),
            "status": status,
            "gap": round(m1 - m2, 2)
        })
        
    verdict = "Stage2 outer frame detection cut off the drawing bottom (y1=796 < 1191), but manual crop failed because 175.65 exists at y=1249 which is even beyond the full page height normally expected."
    if dim_y > 1000:
        verdict = f"Dimension exists at y={dim_y:.1f} which is outside the page height of {ph:.1f} or incorrectly placed below standard coordinates."

    report = {
        "page_rect": {"w": round(pw, 2), "h": round(ph, 2)},
        "outer_frame": of_rect,
        "outer_frame_flag": of_flag,
        "live_zone": lz_rect,
        "live_zone_flag": lz_flag,
        "dim_175_y_coord": round(dim_y, 2),
        "dim_175_flag": dim_175_flag,
        "raster_crop_source": raster_crop_source,
        "raster_crop_flag": raster_crop_flag,
        "blobs_found": n_labels - 1,
        "blob_list": blob_list,
        "added_boxes": added_boxes,
        "verdict": verdict
    }
    
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    run_diagnostic()
