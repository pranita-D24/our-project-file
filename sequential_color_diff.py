import os
import sys
import fitz
import numpy as np
import cv2
from scipy.spatial import KDTree

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DPI = 150
TOLERANCE_PT = 50.0  # distance for matching regions
AREA_DIFF_THRESHOLD = 0.30
MIN_REGION_AREA_PX = 2000
TITLE_BLOCK_Y_LIMIT = 0.85 

COLORS = {
    "ADDED": (0, 200, 0),  # GREEN
}

# ═══════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════

def boxes_near(b1, b2, dist_pt):
    # b = (x0, y0, x1, y1) in pt
    # Expand b1 by dist_pt in all directions
    eb1 = (b1[0] - dist_pt, b1[1] - dist_pt, b1[2] + dist_pt, b1[3] + dist_pt)
    # Check overlap
    return not (eb1[2] < b2[0] or eb1[0] > b2[2] or eb1[3] < b2[1] or eb1[1] > b2[3])

def cluster_into_regions(primitives, cluster_dist_pt):
    if not primitives:
        return []
        
    boxes = [list(p.bbox) for p in primitives]
    
    # Simple iterative merge
    changed = True
    while changed:
        changed = False
        new_boxes = []
        used = set()
        for i in range(len(boxes)):
            if i in used: continue
            curr = list(boxes[i])
            for j in range(i + 1, len(boxes)):
                if j in used: continue
                if boxes_near(curr, boxes[j], cluster_dist_pt):
                    curr[0] = min(curr[0], boxes[j][0])
                    curr[1] = min(curr[1], boxes[j][1])
                    curr[2] = max(curr[2], boxes[j][2])
                    curr[3] = max(curr[3], boxes[j][3])
                    used.add(j)
                    changed = True
            new_boxes.append(tuple(curr))
            used.add(i)
        boxes = new_boxes
        
    regions = []
    for b in boxes:
        centroid = [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]
        area = (b[2] - b[0]) * (b[3] - b[1])
        regions.append({"bbox": b, "centroid": centroid, "area": area})
        
    return regions

# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

class Primitive:
    def __init__(self, bbox):
        self.bbox = bbox

class SequentialColorDiffEngine:
    def __init__(self):
        pass

    def extract_primitives(self, page):
        primitives = []
        page_h = page.rect.height
        page_w = page.rect.width
        max_area = page_w * page_h * 0.5 # Ignore anything larger than 50% page area (frames)
        
        num_text = 0
        num_path = 0
        
        # 1. Extract Text
        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            if "lines" not in block: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    bbox = span["bbox"]
                    if bbox[1] > page_h * TITLE_BLOCK_Y_LIMIT:
                        continue
                    if span["text"].strip():
                        primitives.append(Primitive(bbox))
                        num_text += 1
        
        # 2. Extract Paths
        for path in page.get_drawings():
            bbox = path["rect"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # Rule: Skip massive frames and tiny noise
            if area > max_area or area < 10:
                continue
            if bbox[1] > page_h * TITLE_BLOCK_Y_LIMIT:
                continue
            primitives.append(Primitive(bbox))
            num_path += 1
            
        print(f"Extraction Summary: {len(primitives)} total (Text: {num_text}, Path: {num_path})")
        return primitives

    def compare(self, page1, page2, cluster_dist=20.0):
        v1_prims = self.extract_primitives(page1)
        v2_prims = self.extract_primitives(page2)
        
        # STEP 1 - CLUSTER
        v1_regions = cluster_into_regions(v1_prims, cluster_dist)
        v2_regions = cluster_into_regions(v2_prims, cluster_dist)
        
        print(f"Clustering (dist={cluster_dist}pt): V1={len(v1_regions)} regions, V2={len(v2_regions)} regions")
        
        # STEP 2 - MATCH REGIONS
        if not v1_regions:
            added_candidates = v2_regions
        else:
            v1_centroids = np.array([r["centroid"] for r in v1_regions])
            tree = KDTree(v1_centroids)
            
            added_candidates = []
            num_unmatched_pre_filter = 0
            for r2 in v2_regions:
                dist, idx1 = tree.query(r2["centroid"])
                is_match = False
                if dist < TOLERANCE_PT:
                    r1 = v1_regions[idx1]
                    area_diff = abs(r2["area"] - r1["area"]) / (max(r1["area"], r2["area"]) + 1e-6)
                    if area_diff < AREA_DIFF_THRESHOLD:
                        is_match = True
                
                if not is_match:
                    num_unmatched_pre_filter += 1
                    # Minimum area check (2000px)
                    scale = DPI / 72.0
                    px_w = (r2["bbox"][2] - r2["bbox"][0]) * scale
                    px_h = (r2["bbox"][3] - r2["bbox"][1]) * scale
                    if (px_w * px_h) < MIN_REGION_AREA_PX:
                        continue
                    added_candidates.append(r2)
            
            print(f"Debug: {num_unmatched_pre_filter} unmatched V2 regions found BEFORE area filter.")
                
        return added_candidates

    def render_output(self, page1, page2, added_regions, output_path):
        mat = fitz.Matrix(DPI/72, DPI/72)
        pix1 = page1.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        pix2 = page2.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        
        img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.h, pix1.w, 3)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        
        img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, 3)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        analysis = img2.copy()
        scale = DPI / 72.0
        
        for i, r in enumerate(added_regions):
            b = r["bbox"]
            x0, y0, x1, y1 = [int(c * scale) for c in b]
            w, h = x1 - x0, y1 - y0
            print(f"  Region {i+1}: x={x0} y={y0} w={w} h={h}")
            cv2.rectangle(analysis, (x0, y0), (x1, y1), COLORS["ADDED"], 3)
            cv2.putText(analysis, "ADDED", (x0, max(y0-10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["ADDED"], 3)

        target_h = 1000
        def fit(img):
            r = target_h / img.shape[0]
            return cv2.resize(img, (int(img.shape[1]*r), target_h), interpolation=cv2.INTER_AREA)
            
        p1, p2, p3 = fit(img1), fit(img2), fit(analysis)
        panel = np.hstack([p1, p2, p3])
        
        bar = np.ones((70, panel.shape[1], 3), dtype=np.uint8) * 255
        h_labels = ["V1 (ORIGINAL)", "V2 (REVISION)", "ANALYSIS"]
        for i, txt in enumerate(h_labels):
            cv2.putText(bar, txt, (i*p1.shape[1]+20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 3)
            
        final = np.vstack([bar, panel])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, final)
        print(f"Final Report: ADDED: {len(added_regions)} regions found.")

def run(path1, path2):
    doc1 = fitz.open(path1)
    doc2 = fitz.open(path2)
    p1, p2 = doc1[0], doc2[0]
    engine = SequentialColorDiffEngine()
    
    # Initial run
    added = engine.compare(p1, p2, cluster_dist=20.0)
    
    # Tuning loop
    if len(added) > 10:
        print("Tuning: too many regions found, increasing cluster distance to 40pt...")
        added = engine.compare(p1, p2, cluster_dist=40.0)
        
    engine.render_output(p1, p2, added, "visuals/sequential_diff_report.png")
    doc1.close(); doc2.close()

if __name__ == "__main__":
    v1 = sys.argv[1] if len(sys.argv) > 1 else r"Drawings\PRV73B124138 - Copy.PDF"
    v2 = sys.argv[2] if len(sys.argv) > 2 else r"Drawings\PRV73B124138.PDF"
    run(v1, v2)
