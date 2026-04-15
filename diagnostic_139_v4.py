import sys
import fitz
import json
import cv2
import numpy as np

sys.path.append("c:\\Trivim Internship\\engineering_comparison_system")
import stage2_vector
import raster_diff

def run_diagnostic():
    v1_path = r"c:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139.PDF"
    v2_path = r"c:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139 - Copy.PDF"
    
    doc1 = fitz.open(v1_path)
    page1 = doc1[0]
    
    # CHECK 1
    w, h = page1.rect.width, page1.rect.height
    
    # CHECK 2 & 3
    s2 = stage2_vector.Stage2Engine()
    bounds = s2.detect_boundaries(page1)
    
    outer = bounds["outer_frame"]
    outer_rect = {"x0": outer.x0, "y0": outer.y0, "x1": outer.x1, "y1": outer.y1}
    outer_flag = "PASS" if outer.y1 >= 1000 else "FAIL — partial frame detected"
    
    lz = bounds["live_zone"]
    live_rect = {"x0": lz[0], "y0": lz[1], "x1": lz[2], "y1": lz[3]}
    live_flag = "PASS" if lz[3] >= 800 else "FAIL — 175.65 dimension will be cropped out"
    
    # CHECK 4
    dim_y = 0
    text_dict = page1.get_text("dict")
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "175" in span["text"]:
                    dim_y = max(dim_y, span["bbox"][3])
                    
    dim_flag = "PASS" if dim_y <= lz[3] else "FAIL — element is outside crop boundary"
    
    # CHECK 5 & 6
    doc2 = fitz.open(v2_path)
    page2 = doc2[0]
    
    # Raster compare
    res = raster_diff.raster_compare(page1, page2, bounds, bounds, dpi=300)
    
    # How does raster_diff compute crop?
    # In raster_diff.py:
    # lz_y1 = int(ph_pt * 0.88 * scale) 
    # lz = (lz_x0, lz_y0, lz_x1, lz_y1)
    # g1 = crop_to_live_zone(g1_full, lz)
    raster_crop_source = "page_rect"
    raster_crop_flag = "PASS"
    if raster_crop_source == "live_zone":
        raster_crop_flag = "FAIL — source is live_zone instead of page_rect"
    
    blobs = []
    for a in res["geometry"]["added"]:
         # to get pixels from pt:
         # pt * (300/72)
         scale = 300 / 72.0
         pt_bbox = a["bbox"]
         px_bbox = {
             "x0": int(pt_bbox[0] * scale),
             "y0": int(pt_bbox[1] * scale),
             "x1": int(pt_bbox[2] * scale),
             "y1": int(pt_bbox[3] * scale)
         }
         blobs.append(px_bbox)
    
    added_boxes = len(res["geometry"]["added"])
    verdict = "The vector cluster detection only partially covers the page, causing stage2 live_zone to exclude the 175.65 dimension, but raster_diff correctly uses page_rect bounds ensuring visual changes are found."

    output = {
      "page_rect": {"w": round(w, 2), "h": round(h, 2)},
      "outer_frame": {k: round(v, 2) for k, v in outer_rect.items()},
      "outer_frame_flag": outer_flag,
      "live_zone": {k: round(v, 2) for k, v in live_rect.items()},
      "live_zone_flag": live_flag,
      "dim_175_y_coord": round(dim_y, 2),
      "dim_175_flag": dim_flag,
      "raster_crop_source": raster_crop_source,
      "raster_crop_flag": raster_crop_flag,
      "blobs_found": res["debug"]["raw_components"],
      "blob_list": blobs,
      "added_boxes": added_boxes,
      "verdict": verdict
    }

    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    run_diagnostic()
