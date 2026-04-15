import os
import json
import re
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set

import fitz
import numpy as np
import cv2
import yaml
from scipy.spatial import KDTree
import stage3_balloons
from sklearn.cluster import DBSCAN
import unicodedata

SYMBOL_MAP = {
    "\u00f8": "⌀",   # ø → diameter
    "\u2205": "⌀",   # ∅ → diameter
    "\u00b0": "°",   # degree
    "\u00b1": "±",   # plus-minus
    "\ufffd": "?",   # replacement char
    "\x00":   "",    # null byte
}

def normalize_cad_text(raw: str) -> str:
    """Normalize CAD-specific unicode symbols and strip garbage bytes."""
    result = []
    for ch in raw:
        if ch in SYMBOL_MAP:
            result.append(SYMBOL_MAP[ch])
        elif unicodedata.category(ch) in ("Cc", "Cs", "Co"):
            # Control chars, surrogates, private use → drop
            continue
        else:
            result.append(ch)
    return "".join(result).strip()

# ═══════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════

class VectorEntity:
    def __init__(self, bbox: Tuple[float, float, float, float], entity_type: str):
        self.bbox = bbox # [x0, y0, x1, y1]
        self.type = entity_type
        self.centroid = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.id = f"{entity_type}_{int(time.time() * 1000) % 100000}_{os.urandom(2).hex()}"

class DimensionEntity(VectorEntity):
    def __init__(self, bbox: Tuple[float, float, float, float], value: str, orientation: str, median_height: float = 0.0):
        super().__init__(bbox, "dimension")
        self.value = value
        self.orientation = orientation # "horizontal" or "vertical"
        self.median_height = median_height

class GeometryEntity(VectorEntity):
    def __init__(self, bbox: Tuple[float, float, float, float], geom_type: str, data: Dict[str, Any]):
        super().__init__(bbox, geom_type)
        self.data = data

# ═══════════════════════════════════════════════════════════
# STAGE 2 ENGINE
# ═══════════════════════════════════════════════════════════

class Stage2Engine:
    def __init__(self, config_path: str = "pipeline_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.s2_cfg = self.config["stage2"]
        
        self.logger = logging.getLogger("Stage2")
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            formatter = logging.Formatter("[STAGE2] [%(drawing_id)s] %(message)s")
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)
        self.logger.setLevel(logging.INFO)

    def log(self, drawing_id: str, msg: str, level=logging.INFO):
        self.logger.log(level, msg, extra={"drawing_id": drawing_id})

    # -------------------------------------------------------------------------
    # STEP 1: DYNAMIC FRAME & EXCLUSION DETECTION
    # -------------------------------------------------------------------------
    
    def detect_boundaries(self, page: fitz.Page) -> Dict[str, Any]:
        """Hierarchical Cluster Detection: Outermost container or largest area."""
        clusters = page.cluster_drawings()
        if not clusters:
            return {"outer_frame": page.rect, "right_edge": page.rect.width, "bottom_edge": page.rect.height}
        
        frame_cluster = None
        # Logic: If a cluster contains most others, it's the frame/boundary box
        for cluster in clusters:
            contained = sum(1 for c in clusters if cluster.contains(c))
            if contained >= len(clusters) * 0.6:
                frame_cluster = cluster
                break
        
        # Fallback: Largest by area
        if frame_cluster is None:
            frame_cluster = max(clusters, key=lambda r: r.width * r.height)
            
        PAGE_H = page.rect.height
        if frame_cluster.y1 < (PAGE_H * 0.85):
            frame_cluster = page.rect
            
        f_x0, f_y0, f_x1, f_y1 = frame_cluster.x0, frame_cluster.y0, frame_cluster.x1, frame_cluster.y1
        f_w = f_x1 - f_x0
        f_h = f_y1 - f_y0
        
        # USER FIX 2+A: Compute strict live zone inset
        live_zone = (
            f_x0 + (f_w * 0.05), # left boundary
            f_y0 + (f_h * 0.08), # top boundary (FIX A: Increased to 8% to block grid headers)
            f_x1 - (f_w * 0.05), # right boundary
            f_y1 - (f_h * 0.15)  # bottom boundary
        )
            
        return {
            "outer_frame": frame_cluster,
            "right_edge": frame_cluster.x1,
            "bottom_edge": frame_cluster.y1,
            "live_zone": live_zone
        }

    # -------------------------------------------------------------------------
    # STEP 2-6: PIPELINE EXTRACTION
    # -------------------------------------------------------------------------

    def is_rotated(self, bbox: Tuple[float, float, float, float]) -> bool:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return h > w * 1.5

    def extract_page_data(self, page: fitz.Page, drawing_id: str) -> Dict[str, Any]:
        """Strict Order: Frame -> Exclude -> Label -> Stats -> Group -> Regex"""
        
        # 1. Detect Frame
        bounds = self.detect_boundaries(page)
        f = bounds["outer_frame"]
        
        text_dict = page.get_text("dict")
        drawings = page.get_drawings()
        
        # 2. Filter Spans (BBoxes & Zones)
        surviving_spans = []
        all_heights = []
        
        for b_idx, block in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block: continue
            for l_idx, line in enumerate(block["lines"]):
                for s_idx, span in enumerate(line["spans"]):
                    txt = normalize_cad_text(span["text"])
                    if not txt: continue
                    
                    b = span["bbox"]
                    # Rule: Title Block Exclusion (Bottom 20% of page)
                    if b[1] > (page.rect.height * 0.80):
                        continue
                        
                    # Rule: Outside Frame?
                    if b[0] < f[0]-2 or b[2] > f[2]+2 or b[1] < f[1]-2 or b[3] > f[3]+2:
                        continue
                        
                    # Rule: Grid Labels? (Size ~13.9, len <= 2, near edge)
                    dist_to_edge = min(abs(b[0]-f[0]), abs(b[2]-f[2]), abs(b[1]-f[1]), abs(b[3]-f[3]))
                    if span["size"] > 12 and len(txt) <= 2 and dist_to_edge < 15:
                        continue
                        
                    # Index check for Step 3? (Wait, Step 3 needs the full text_dict)
                    pass
                        
                    # Potential candidate
                    surviving_spans.append({
                        "text": txt,
                        "bbox": b,
                        "size": span["size"],
                        "centroid": [(b[0]+b[2])/2, (b[1]+b[3])/2],
                        "idx": (b_idx, l_idx, s_idx)
                    })
                    all_heights.append(b[3] - b[1])

        if not surviving_spans:
            return {"balloons": [], "dimensions": [], "geometry": []}

        # --- STAGE 3: BALLOON DETECTION ---
        balloon_results = stage3_balloons.detect_balloons(page, text_dict, drawing_id)
        ignored_text_indices = balloon_results["text_span_indices"]
        ignored_path_indices = balloon_results["path_indices"]
        
        # Filter surviving_spans to remove identified balloons
        clean_spans = [s for s in surviving_spans if s["idx"] not in ignored_text_indices]
        surviving_spans = clean_spans

        # --- STAGE 4: DBSCAN DIMENSION GROUPING ---
        # 1. Identify "Dimension-Like" Spans
        dim_regex = re.compile(self.s2_cfg.get("dim_pattern", r'[ØRr⌀±~+\-\xad]?\d+(\.\d+)?([\xad±\-+]\d+(\.\d+)?)*'))
        dim_spans = [s for s in surviving_spans if dim_regex.search(s["text"])]
        
        balloons = balloon_results["locations"]
        if not dim_spans:
            dimensions = []
        else:
            # 2. Dynamic Epsilon: Median Height * 1.5
            median_h = np.median([s["bbox"][3] - s["bbox"][1] for s in dim_spans])
            eps = max(10, median_h * 1.5) # Floor at 10pt
            
            # 3. Spatial Clustering
            coords = np.array([s["centroid"] for s in dim_spans])
            db = DBSCAN(eps=eps, min_samples=1).fit(coords)
            labels = db.labels_
            
            dimensions = []
            for label in set(labels):
                cluster = [dim_spans[i] for i, l in enumerate(labels) if l == label]
                # Sort for logical text order
                cluster.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
                val = "".join(s["text"] for s in cluster)
                
                x0 = min(s["bbox"][0] for s in cluster)
                y0 = min(s["bbox"][1] for s in cluster)
                x1 = max(s["bbox"][2] for s in cluster)
                y1 = max(s["bbox"][3] for s in cluster)
                
                orientation = "vertical" if self.is_rotated((x0, y0, x1, y1)) else "horizontal"
                dimensions.append(DimensionEntity((x0, y0, x1, y1), val, orientation, median_height=median_h))

        # 6. Geometry Extraction (Filtered by Frame & Balloons)
        geometry = []
        for p_idx, d in enumerate(drawings):
            # Rule: Balloon?
            if p_idx in ignored_path_indices:
                continue
                
            # Rule: Outside Frame? (Check center)
            r = d["rect"]
            cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
            if not (f[0]-2 <= cx <= f[2]+2 and f[1]-2 <= cy <= f[3]+2):
                continue
                
            # Exclude Title Block (Dynamic: Rightmost 15% of frame)
            if cx > f[0] + f.width*0.7 and cy > f[1] + f.height*0.6:
                continue
                
            geom_type = "structural"
            w = d.get("width")
            if w is not None and w < 0.5: geom_type = "thin_line"
            geometry.append(GeometryEntity(r, geom_type, {"width": d["width"]}))

        return {
            "balloons": balloons,
            "balloons_count": len(balloons),
            "dimensions": dimensions,
            "geometry": geometry
        }

    # -------------------------------------------------------------------------
    # COMPARISON & REPORTING
    # -------------------------------------------------------------------------

    def match_entities(self, v1_list: List[VectorEntity], v2_list: List[VectorEntity], tol: float):
        if not v1_list or not v2_list:
            return [], v1_list, v2_list
        v1_centroids = np.array([e.centroid for e in v1_list])
        v2_centroids = np.array([e.centroid for e in v2_list])
        tree = KDTree(v2_centroids)
        matches, v2_m, v1_m = [], set(), set()
        for i, c1 in enumerate(v1_centroids):
            dist, idx = tree.query(c1)
            if dist < tol and idx not in v2_m:
                matches.append((v1_list[i], v2_list[idx]))
                v2_m.add(idx)
                v1_m.add(i)
        rem = [v1_list[i] for i in range(len(v1_list)) if i not in v1_m]
        add = [v2_list[i] for i in range(len(v2_list)) if i not in v2_m]
        return matches, rem, add

    def process_pair(self, path_v1: str, path_v2: str, drawing_id: str, output_root: str):
        self.log(drawing_id, "Advanced Stage 2 Processing Started")
        doc1 = fitz.open(path_v1)
        doc2 = fitz.open(path_v2)
        report = {"drawing_id": drawing_id, "pages": []}
        
        for p_idx in range(min(len(doc1), len(doc2))):
            d1 = self.extract_page_data(doc1[p_idx], drawing_id)
            d2 = self.extract_page_data(doc2[p_idx], drawing_id)
            
            # Dimension Match
            tol = self.s2_cfg["centroid_match_tolerance_pt"]
            m, rem, add = self.match_entities(d1["dimensions"], d2["dimensions"], tol)
            mod = [{"from": p[0].value, "to": p[1].value, "bbox": list(p[1].bbox)} for p in m if p[0].value != p[1].value]
            
            # Geometry Match
            gm, grem, gadd = self.match_entities(d1["geometry"], d2["geometry"], tol)
            resized = []
            for g1, g2 in gm:
                # Use Diagonal length as size proxy
                s1 = ((g1.bbox[2]-g1.bbox[0])**2 + (g1.bbox[3]-g1.bbox[1])**2)**0.5
                s2 = ((g2.bbox[2]-g2.bbox[0])**2 + (g2.bbox[3]-g2.bbox[1])**2)**0.5
                if abs(s1-s2) > self.s2_cfg["resize_threshold_pt"]:
                    resized.append({"from_bbox": list(g1.bbox), "to_bbox": list(g2.bbox)})

            page_dict = {
                "page": p_idx,
                "balloons_ignored": d2.get("balloons_count", 0),
                "dimensions": {"added": [list(e.bbox) for e in add], "removed": [list(e.bbox) for e in rem], "modified": mod},
                "geometry": {"added": [list(e.bbox) for e in gadd], "removed": [list(e.bbox) for e in grem], "resized": resized}
            }
            report["pages"].append(page_dict)

        doc1.close(); doc2.close()
        out_json = Path(output_root) / f"{drawing_id}_stage2.json"
        with open(out_json, "w") as f: json.dump(report, f, indent=2)
        return report

if __name__ == "__main__":
    pass
