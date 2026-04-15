import cv2
import numpy as np
import os
from cad_audit.extraction.extractor import extract_drawing
from cad_audit.comparison.title_filter import filter_structural
from cad_audit.comparison.matcher import compare_spans
from cad_audit.comparison.verdict import compute_verdict
from cad_audit.reporting.marker_draw import draw_markers

def run_test():
    # Paths (Real drawing vs its copy/revision)
    v1_path = "Drawings/PRV73B124143.PDF"
    v2_path = "Drawings/PRV73B124143 - Copy.PDF"
    output_path = "cad_audit/test_marker_check.png"
    
    print(f"--- CAD Audit Foundation Test (Hardened Matcher) ---")
    
    # 1. Extraction (300 DPI coordinate locked)
    print(f"[1/5] Extracting V1...")
    v1_data = extract_drawing(v1_path)[0]
    print(f"[1/5] Extracting V2...")
    v2_data = extract_drawing(v2_path)[0]
    
    # 2. Filtering
    print(f"[2/5] Filtering Administrative Zones...")
    v1_spans = filter_structural(v1_data["spans"])
    v2_spans = filter_structural(v2_data["spans"])
    
    # 3. SPATIAL MATCHING (The Fix)
    print(f"[3/5] Performing KDTree Spatial Matching...")
    added, removed = compare_spans(v1_spans, v2_spans, tolerance_px=20.0)
    
    # 4. Verdict
    print(f"[4/5] Computing Verdict...")
    verdict = compute_verdict(len(added), len(removed))
    print(f"      Verdict: {verdict} (Add: {len(added)}, Rem: {len(removed)})")
    
    # 5. Reporting (PIL-based)
    print(f"[5/5] Generating Visual Report...")
    pix = v2_data["image"]
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    report_img = draw_markers(img, added, removed)
    
    # Save
    cv2.imwrite(output_path, report_img)
    print(f"--- TEST COMPLETE: {output_path} ---")

if __name__ == "__main__":
    run_test()
