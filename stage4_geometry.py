import fitz
import numpy as np
import logging
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from dataclasses import dataclass, field

logger = logging.getLogger("Stage4")

@dataclass
class StructuralEntity:
    type: str # line, arc, circle, rect, polyline
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float] # [x0, y0, x1, y1]
    length: float = 0.0
    radius: float = 0.0
    area: float = 0.0
    stroke_width: float = 0.0
    op_signature: str = ""
    original_idx: int = -1

class Stage4Engine:
    def __init__(self):
        self.logger = logger

    @staticmethod
    def _bbox_area(bbox):
        """Compute area of a bbox [x0, y0, x1, y1] (tuple or list)."""
        if not bbox:
            return 0.0
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        return 0.0

    def get_invariant_signature(self, items: list, p_type: str) -> str:
        """
        Generates a structural signature invariant to draw order and direction.
        """
        if not items: return ""
        
        if p_type == "polyline":
            segments = []
            for it in items:
                if it[0] == "l":
                    # Store as sorted coords to handle direction-independence
                    # Rounding to 0.5pt to handle minor jitter in vector generation
                    p1 = (round(it[1].x * 2) / 2, round(it[1].y * 2) / 2)
                    p2 = (round(it[2].x * 2) / 2, round(it[2].y * 2) / 2)
                    segments.append(tuple(sorted([p1, p2])))
            segments.sort()
            # Signature is based on segment lengths (rounded) to be scale-aware but jitter-resistant
            lengths = []
            for s in segments:
                L = ( (s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2 )**0.5
                lengths.append(f"{L:.1f}")
            return f"poly_{len(segments)}_" + "_".join(lengths)
        
        ops = sorted([it[0] for it in items])
        return "".join(ops)

    def normalize_path(self, p: dict, idx: int, sq_range: Tuple[float, float]) -> Optional[StructuralEntity]:
        items = p.get("items", [])
        r = p["rect"]
        cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
        area = r.width * r.height
        
        # Determine internal SVG/vector type
        if all(it[0] == "re" for it in items):
            p_type = "rect"
        elif all(it[0] in ("c", "qu") for it in items):
            aspect = r.width / r.height if r.height > 0 else 1.0
            p_type = "circle" if sq_range[0] <= aspect <= sq_range[1] else "arc"
        elif all(it[0] == "l" for it in items):
            p_type = "line" if len(items) == 1 else "polyline"
        else:
            p_type = "path" # Match for SPLINE / PATH
            
        # FIX 1: ENTITY TYPE WHITELIST
        WHITELIST = {"path", "circle", "arc", "rect", "polyline", "line"}
        
        if p_type not in WHITELIST:
            return None
            
        # RECT Constraint: Only when aspect ratio > 2.0 or area > 1000
        # OVERRIDE: exempt RECT primitives whose aspect ratio exceeds 3.0 regardless of area threshold.
        if p_type == "rect":
            aspect_ratio = max(r.width/r.height, r.height/r.width) if r.height > 0 and r.width > 0 else 0
            if aspect_ratio > 3.0:
                pass # Exempted narrow rect (Slot)
            elif not (aspect_ratio > 2.0 or area > 1000.0):
                if 200 < cx < 500 and 300 < cy < 600:
                    print(f"DEBUG: REJECTED BY RECT FILTER: {r} (Area={area}, Aspect={aspect_ratio})")
                return None
        
        op_sig = self.get_invariant_signature(items, p_type)
        w_val = p.get("width")
        stroke_width = float(w_val) if w_val is not None else 0.5
        
        # Calculate cumulative length across all segments
        cumul_len = 0.0
        for it in items:
            if it[0] == "l":
                cumul_len += it[1].distance_to(it[2])
            elif it[0] == "c":
                # Approximate bezier curve length via control points
                cumul_len += it[1].distance_to(it[2]) + it[2].distance_to(it[3]) + it[3].distance_to(it[4])

        # FIX C: Path Fragmentation Filter
        # Dimension tick marks and leaders usually parse as short paths. Discard if total length < 15.0pt.
        if p_type in ("path", "polyline", "line"):
            if cumul_len < 15.0:
                return None

        return StructuralEntity(
            type=p_type,
            centroid=(cx, cy),
            bbox=(r[0], r[1], r[2], r[3]),
            length=cumul_len,
            radius=r.width/2 if p_type in ("circle", "arc") else 0.0,
            area=area,
            stroke_width=stroke_width,
            op_signature=op_sig,
            original_idx=idx
        )

    def calibrate_stroke_cutoff(self, drawings: list, balloon_indices: set) -> float:
        widths = []
        for i, d in enumerate(drawings):
            if i not in balloon_indices:
                w = d.get("width")
                if w is not None:
                    try: widths.append(float(w))
                    except: widths.append(0.5)
                else: widths.append(0.5)
        if not widths: return 0.5
        clean_widths = sorted(list(set(widths)))
        if len(clean_widths) < 2: return clean_widths[0] + 0.05
        for w in clean_widths:
            if w > 0.4: break
            idx = clean_widths.index(w)
            if idx + 1 < len(clean_widths):
                nw = clean_widths[idx+1]
                if nw > w + 0.1: return w + 0.05
        return clean_widths[0] + 0.1

    def is_near_dim_text(self, centroid: Tuple[float, float], dim_spans: List[Any], threshold: float) -> bool:
        if not dim_spans: return False
        for span in dim_spans:
            bbox = span.get("bbox") if isinstance(span, dict) else getattr(span, "bbox", None)
            if bbox is None: continue
            bx, by = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            if ((centroid[0]-bx)**2 + (centroid[1]-by)**2)**0.5 < threshold:
                return True
        return False

    def compute_alignment_offset(self, page1: fitz.Page, page2: fitz.Page, 
                                v1_list: List[StructuralEntity], 
                                v2_list: List[StructuralEntity],
                                bounds_v1: dict, bounds_v2: dict) -> Tuple[float, float, float]:
        # 1. Title Block Bottom-Right Anchor (Fix 3)
        tb1 = bounds_v1.get("title_block_bbox")
        tb2 = bounds_v2.get("title_block_bbox")
        if tb1 and tb2 and tb1 != (0,0,0,0) and tb2 != (0,0,0,0):
            # Anchor coordinate system to the title block bottom-right
            dx_tb = tb1[2] - tb2[2]
            dy_tb = tb1[3] - tb2[3]
            self.logger.info(f"Anchored coordinate system to Title Block Bottom-Right. dx: {dx_tb}, dy: {dy_tb}")
            return dx_tb, dy_tb, 1.0

        t1, t2 = page1.get_text("dict"), page2.get_text("dict")
        anchors1 = {}
        for b in t1.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    txt = s["text"].strip()
                    if len(txt) > 3:
                        anchors1.setdefault(txt, []).append(((s["bbox"][0]+s["bbox"][2])/2, (s["bbox"][1]+s["bbox"][3])/2))
        offsets = []
        for b in t2.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    txt = s["text"].strip()
                    if txt in anchors1:
                        c2 = ((s["bbox"][0]+s["bbox"][2])/2, (s["bbox"][1]+s["bbox"][3])/2)
                        for c1 in anchors1[txt]:
                            dx, dy = c1[0]-c2[0], c1[1]-c2[1]
                            if abs(dx) < 50 and abs(dy) < 50: offsets.append((dx, dy))
        # Structural Anchors
        v1_geom = {}
        for e in v1_list: v1_geom.setdefault((e.type, e.op_signature), []).append(e.centroid)
        for e2 in v2_list:
            sig = (e2.type, e2.op_signature)
            if sig in v1_geom and len(v1_geom[sig]) == 1:
                c1 = v1_geom[sig][0]
                offsets.append((c1[0]-e2.centroid[0], c1[1]-e2.centroid[1]))
        if not offsets: return (0.0, 0.0, 0.5)
        dx, dy = float(np.median([o[0] for o in offsets])), float(np.median([o[1] for o in offsets]))
        resids = [((o[0]-dx)**2 + (o[1]-dy)**2)**0.5 for o in offsets]
        return dx, dy, float(np.percentile(resids, 95)) if resids else 0.5

    def extract_filtered(self, page: fitz.Page, balloon_indices: set, 
                        dim_spans: list, bounds: dict, drawings: list = None) -> List[StructuralEntity]:
        if drawings is None:
            drawings = page.get_drawings()
        stroke_cutoff = self.calibrate_stroke_cutoff(drawings, balloon_indices)
        entities = []
        outer = bounds.get("outer_frame", page.rect)
        
        for i, p in enumerate(drawings):
            if i in balloon_indices: continue
            
            # Filter green annotation balloons by COLOR
            color = p.get("color") or p.get("stroke") or [0,0,0]
            if len(color) == 3:
                r_c, g_c, b_c = color
                if g_c > 0.4 and r_c < 0.3 and b_c < 0.3:
                    continue
            
            r = p.get("rect", fitz.Rect(0,0,0,0))
            cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
            
            # 1. FIX 2: STRICT LIVE ZONE INSET VALIDATION
            live_zone = bounds.get("live_zone")
            if live_zone:
                lz_left, lz_top, lz_right, lz_bottom = live_zone
                if not (lz_left <= cx <= lz_right and lz_top <= cy <= lz_bottom):
                    if 200 < cx < 500 and 300 < cy < 600:
                        print(f"DEBUG: REJECTED BY LIVE ZONE: centroid {cx, cy} (Bounds: {live_zone})")
                    continue
            else:
                if not (outer[0]+5 <= cx <= outer[2]-5 and outer[1]+5 <= cy <= outer[3]-5): continue
            
            # 2. Density-Aware Region Suppression (Title Block, Tables, Bom)
            # Refined Rule: Protect Main View and Left Margins (Fix 3)
            # Title Block: Bottom-Right quadrant but allow anything near the left edge.
            is_title_block = (cx > outer[0] + (outer[2]-outer[0]) * 0.75 and cy > outer[1] + (outer[3]-outer[1]) * 0.65)
            is_bottom_strip = (cy > outer[1] + (outer[3]-outer[1]) * 0.92) # Conservative 8% floor
            
            # Protection: Never exclude items in the leftmost 20% of the usable frame
            is_protected = (cx < outer[0] + (outer[2]-outer[0]) * 0.20)
            
            if (is_title_block or is_bottom_strip) and not is_protected:
                continue

            # 3. Dimension Exclusion
            w = p.get("width")
            w_val = float(w) if w is not None else 0.5
            if w_val <= stroke_cutoff and self.is_near_dim_text((cx, cy), dim_spans, 20.0): continue
            
            ent = self.normalize_path(p, i, (0.9, 1.1))
            if ent is not None:
                entities.append(ent)
        return entities

    def cluster_and_filter_noise(self, entities: List[dict], label: str = "ADDED") -> List[dict]:
        """
        Uses DBSCAN and NN-Texture Analysis to suppress administrative tables.
        Refined Fix: Removed Median-based threshold (Fix for PRV73B124140).
        """
        if len(entities) < 15: return entities # Lower threshold for tiny changes
        
        centroids = np.array([e["centroid"] for e in entities])
        if len(centroids) < 5: return entities
        
        tree = KDTree(centroids)
        nn_dists, _ = tree.query(centroids, k=2)
        nn1 = nn_dists[:, 1]
        
        eps = float(np.percentile(nn1, 90))
        eps = max(min(eps, 45.0), 3.0) 
        
        db = DBSCAN(eps=eps, min_samples=8).fit(centroids) # Tightened min_samples
        labels = db.labels_
        
        suppress_indices = set()
        unique_labels = [l for l in set(labels) if l != -1]
        
        cluster_info = []
        for l in unique_labels:
            indices = np.where(labels == l)[0]
            cluster_centroids = centroids[indices]
            c_tree = KDTree(cluster_centroids)
            c_nn_dists, _ = c_tree.query(cluster_centroids, k=2)
            c_nn1 = c_nn_dists[:, 1]
            mean_nn, std_nn = np.mean(c_nn1), np.std(c_nn1)
            cv_nn = std_nn / mean_nn if mean_nn > 0 else 0.0
            cluster_info.append({"label": l, "indices": indices, "cv": cv_nn, "mean": mean_nn})
            
        if not cluster_info: return entities
        
        # Refined Heuristic: Use Fixed CV + Density Thresholds
        # Table lines are extremely regular (CV < 0.15) and tight (Mean dist < 8pt)
        for c in cluster_info:
            is_extremely_regular = (c["cv"] < 0.12) # Very tight regular grid
            is_dense_grid = (c["cv"] < 0.25 and c["mean"] < 7.0) # Small regular table entries
            
            if is_extremely_regular or is_dense_grid:
                # Safety: Don't suppress if the cluster is very large (could be a complex part)
                # unless it's extremely regular
                if len(c["indices"]) < 100 or is_extremely_regular:
                    self.logger.info(f"Suppressing {label} cluster {c['label']} (size={len(c['indices'])}, cv={c['cv']:.4f}) as Admin Noise.")
                    for idx in c["indices"]:
                        suppress_indices.add(idx)
                    
        return [e for i, e in enumerate(entities) if i not in suppress_indices]

    def compare_pages(self, page1, page2, dim_spans_v1, dim_spans_v2, 
                    balloons_v1, balloons_v2, bounds_v1, bounds_v2,
                    drawings_v1=None, drawings_v2=None):
        v1_list = self.extract_filtered(page1, balloons_v1, dim_spans_v1, bounds_v1, drawings_v1)
        v2_list = self.extract_filtered(page2, balloons_v2, dim_spans_v2, bounds_v2, drawings_v2)
        
        dx, dy, jitter = self.compute_alignment_offset(page1, page2, v1_list, v2_list, bounds_v1, bounds_v2)
        for e in v2_list:
            e.centroid = (e.centroid[0]+dx, e.centroid[1]+dy)
            e.bbox = (e.bbox[0]+dx, e.bbox[1]+dy, e.bbox[2]+dx, e.bbox[3]+dy)
            
        v1_cent = np.array([e.centroid for e in v1_list])
        tree_v1 = KDTree(v1_cent) if len(v1_cent) > 0 else None

        # FIX A: Dynamic centroid tolerance = 1.5% of live zone width
        # Scales correctly regardless of image resolution / DPI
        live_zone_v2 = bounds_v2.get("live_zone")
        if live_zone_v2:
            lz_width = live_zone_v2[2] - live_zone_v2[0]
            lz_height = live_zone_v2[3] - live_zone_v2[1]
            lz_area = lz_width * lz_height
        else:
            lz_width = page2.rect.width
            lz_height = page2.rect.height
            lz_area = lz_width * lz_height
        match_tol = lz_width * 0.015  # 1.5% of live zone width
        match_tol = max(match_tol, 1.0)  # floor at 1pt
        self.logger.info(f"Dynamic match_tol = {match_tol:.2f}pt (lz_width={lz_width:.1f})")

        added, removed, resized, unchanged = [], [], [], 0
        matched_v1 = set()
        
        if tree_v1:
            for i2, e2 in enumerate(v2_list):
                d, i1 = tree_v1.query(e2.centroid)
                e1 = v1_list[i1]

                # FIX B: Aspect ratio hard guard — >15% difference = no match
                w1, h1 = e1.bbox[2]-e1.bbox[0], e1.bbox[3]-e1.bbox[1]
                w2, h2 = e2.bbox[2]-e2.bbox[0], e2.bbox[3]-e2.bbox[1]
                ar1 = w1 / h1 if h1 > 0.01 else 0.0
                ar2 = w2 / h2 if h2 > 0.01 else 0.0
                ar_diff = abs(ar1 - ar2) / max(ar1, ar2, 0.01)
                aspect_ok = ar_diff <= 0.15

                if d < match_tol and e1.type == e2.type and e1.op_signature == e2.op_signature and aspect_ok:
                    if i1 not in matched_v1:
                        matched_v1.add(i1)
                        if abs(w1-w2) > 0.5 or abs(h1-h2) > 0.5:
                            resized.append({"type":e2.type,"centroid":e2.centroid,"from_bbox":e1.bbox,"to_bbox":e2.bbox})
                        else: unchanged += 1
                    else: added.append(asdict(e2))
                else: added.append(asdict(e2))
            for i1, e1 in enumerate(v1_list):
                if i1 not in matched_v1: removed.append(asdict(e1))
        
        # FIX 4: Coordinate Space Validation
        viewBox_x = page1.rect.x0
        viewBox_y = page1.rect.y0
        viewBox_width = page1.rect.width
        viewBox_height = page1.rect.height

        valid_x_min = viewBox_x - (viewBox_width * 0.05)
        valid_x_max = viewBox_x + (viewBox_width * 1.05)
        valid_y_min = viewBox_y - (viewBox_height * 0.05)
        valid_y_max = viewBox_y + (viewBox_height * 1.05)

        def is_valid_centroid(c):
            v = (valid_x_min <= c[0] <= valid_x_max) and (valid_y_min <= c[1] <= valid_y_max)
            if not v and 200 < c[0] < 500 and 300 < c[1] < 600:
                 print(f"DEBUG: REJECTED BY FIX 4 COORDINATE VALIDATOR: {c}")
            return v
            
        added = [a for a in added if is_valid_centroid(a["centroid"])]
        removed = [r for r in removed if is_valid_centroid(r["centroid"])]
        resized = [rs for rs in resized if is_valid_centroid(rs["centroid"])]
        
        # 4. Filter Noise Islands (Tables/BOMs)
        added = self.cluster_and_filter_noise(added, "ADDED")
        removed = self.cluster_and_filter_noise(removed, "REMOVED")

        # FIX C: Minimum size filter — reject ADDED candidates whose bounding box
        # is smaller than 1% of live zone area (kills dimension tick false positives)
        min_area_threshold = lz_area * 0.01
        pre_count = len(added)
        added = [a for a in added if self._bbox_area(a.get("bbox")) >= min_area_threshold]
        if pre_count != len(added):
            self.logger.info(f"Min-area filter removed {pre_count - len(added)} tiny ADDED candidates (threshold={min_area_threshold:.1f}pt²)")

        return { "geometry": { "added": added, "removed": removed, "resized": resized, "unchanged_count": unchanged } }

def asdict(obj):
    if hasattr(obj, "__dict__"): return {k: asdict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)): return [asdict(x) for x in obj]
    if isinstance(obj, dict): return {k: asdict(v) for k, v in obj.items()}
    return obj
