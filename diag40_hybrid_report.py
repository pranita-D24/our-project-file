import os
import sys
import fitz
import numpy as np
import cv2

# CONFIGURATION
DPI = 150
COLORS = {"ADDED": (0, 200, 0)} # GREEN

def run_hybrid_diff_diag40():
    v1_path = r"Drawings\PRV73B124140 - Copy.PDF"
    v2_path = r"Drawings\PRV73B124140.PDF"
    output_path = "visuals/sequential_diff_report.png"

    # 1. Pixel-Diff coordinates (found in diagnostic)
    # 3D: x=1321 y=62 w=1091 h=308
    # Note: x=1453 y=62 w=959 h=21 
    # (Wait, they overlap? Let's treat them as two logical entities with padding)
    
    pad = int(10 * (DPI / 72.0))
    added_boxes = [
        {"x": 1321 - pad, "y": 62 - pad, "w": 1091 + 2*pad, "h": 308 + 2*pad},
        {"x": 1453 - pad, "y": 62 - pad, "w": 959 + 2*pad, "h": 21 + 2*pad}
    ]
    
    # 2. Render panels
    doc1 = fitz.open(v1_path)
    doc2 = fitz.open(v2_path)
    mat = fitz.Matrix(DPI/72, DPI/72)
    
    pix1 = doc1[0].get_pixmap(matrix=mat)
    pix2 = doc2[0].get_pixmap(matrix=mat)
    
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.h, pix1.w, 3)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, 3)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    analysis = img2.copy()
    
    for box in added_boxes:
        cv2.rectangle(analysis, (box["x"], box["y"]), (box["x"]+box["w"], box["y"]+box["h"]), COLORS["ADDED"], 4)
        cv2.putText(analysis, "ADDED", (box["x"], box["y"]-10 if box["y"] > 30 else box["y"]+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["ADDED"], 4)

    # Horizontal Stack
    target_h = 1000
    def fit(img):
        r = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1]*r), target_h), interpolation=cv2.INTER_AREA)
    
    p1, p2, p3 = fit(img1), fit(img2), fit(analysis)
    panel = np.hstack([p1, p2, p3])
    
    bar = np.ones((70, panel.shape[1], 3), dtype=np.uint8) * 255
    h_labels = ["V1 (ORIGINAL)", "V2 (REVISION)", "ANALYSIS | ADDED: 2"]
    for i, txt in enumerate(h_labels):
        cv2.putText(bar, txt, (i*p1.shape[1]+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 3)
        
    final = np.vstack([bar, panel])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)
    print(f"Hybrid Report Saved: {output_path}")

if __name__ == '__main__':
    run_hybrid_diff_diag40()
