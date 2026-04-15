import fitz
import numpy as np
import cv2
import os
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

class Stage1AddedDetector:
    def __init__(self, tol_distance=20.0, tol_fingerprint=0.10):
        self.tol_dist = tol_distance
        self.tol_fp = tol_fingerprint
        
    def _get_live_zone(self, page):
        # Step 2: Define the Live Zone
        # Detect outermost drawing border/frame
        clusters = page.cluster_drawings()
        if not clusters:
            f = page.rect
        else:
            f = max(clusters, key=lambda r: r.width * r.height)
            
        fw, fh = f.width, f.height
        return {
            "x0": f.x0 + fw * 0.02,
            "y0": f.y0 + fh * 0.08,
            "x1": f.x1 - fw * 0.02,
            "y1": f.y1 - fh * 0.10
        }

    def _parse_primitives(self, page, lz):
        # Step 1: Parse & Extract Primitives
        allowed_types = {"line", "polyline", "rect", "circle", "arc", "path"}
        elements = []
        
        drawings = page.get_drawings()
        for idx, d in enumerate(drawings):
            r = d.get("rect")
            if not r: continue
            
            items = d.get("items", [])
            if not items: continue

            # Determine generic type
            if all(it[0] == "re" for it in items):
                p_type = "rect"
            elif all(it[0] in ("c", "qu") for it in items):
                aspect = r.width / r.height if r.height > 0 else 1.0
                p_type = "circle" if 0.9 <= aspect <= 1.1 else "arc"
            elif all(it[0] == "l" for it in items):
                p_type = "line" if len(items) == 1 else "polyline"
            else:
                p_type = "path"

            if p_type not in allowed_types:
                continue
                
            # Filter outside live zone
            cx = (r[0] + r[2]) / 2
            cy = (r[1] + r[3]) / 2
            if not (lz["x0"] <= cx <= lz["x1"] and lz["y0"] <= cy <= lz["y1"]):
                continue

            # Length calculation for Short Path rejection
            cumul_len = 0.0
            perimeter = 0.0
            for it in items:
                if it[0] == "l":
                    L = it[1].distance_to(it[2])
                    cumul_len += L
                    perimeter += L
                elif it[0] == "c":
                    L = it[1].distance_to(it[2]) + it[2].distance_to(it[3]) + it[3].distance_to(it[4])
                    cumul_len += L
                    perimeter += L
                elif it[0] == "re":
                    L = (it[1].width + it[1].height) * 2
                    cumul_len += L
                    perimeter += L
                    
            if cumul_len < 15.0:
                continue

            # Reject pure hatch fills (without structural stroke lines)
            op_type = d.get("type", "s")
            if "f" in op_type and "s" not in op_type:
                continue

            # Step 3: Build Shape Signatures
            area = r.width * r.height
            aspect_ratio = max(r.width/r.height, r.height/r.width) if r.height > 0 and r.width > 0 else 0
            
            fingerprint = (round(area, 1), round(aspect_ratio, 1), round(perimeter, 1))

            elements.append({
                "type": p_type,
                "centroid": (cx, cy),
                "bbox": [r[0], r[1], r[2], r[3]],
                "width": r.width,
                "height": r.height,
                "area": area,
                "aspect_ratio": aspect_ratio,
                "perimeter": perimeter,
                "fingerprint": fingerprint
            })
            
        return elements

    def detect_added(self, path_v1, path_v2):
        doc1 = fitz.open(path_v1)
        doc2 = fitz.open(path_v2)
        p1, p2 = doc1[0], doc2[0]

        lz1 = self._get_live_zone(p1)
        lz2 = self._get_live_zone(p2)

        v1_elems = self._parse_primitives(p1, lz1)
        v2_elems = self._parse_primitives(p2, lz2)

        candidates = []
        
        # Dynamic parameters based on Live Zone width and area
        lz_width = lz2["x1"] - lz2["x0"]
        lz_area = lz_width * (lz2["y1"] - lz2["y0"])
        dynamic_centroid_tol = lz_width * 0.015
        min_cluster_area = lz_area * 0.01
        
        # Step 4: Spatial Matching
        # V1 tree for fast spatial lookup
        if not v1_elems:
            candidates = v2_elems
        else:
            v1_cents = np.array([e["centroid"] for e in v1_elems])
            tree = KDTree(v1_cents)
            
            matched_v1_indices = set()

            for e2 in v2_elems:
                # Find all neighbors within dynamic tolerance directly tied to zone width
                indices = tree.query_ball_point(e2["centroid"], dynamic_centroid_tol)
                
                match_found = False
                for i1 in indices:
                    if i1 in matched_v1_indices: continue
                    e1 = v1_elems[i1]
                    
                    if e1["type"] != e2["type"]: continue
                    
                    # Area fingerprint tolerance check
                    if abs(e1["area"] - e2["area"]) > max(e1["area"], e2["area"]) * self.tol_fp:
                        continue
                        
                    # Aspect Ratio guard hard check (Variance > 15% = NO MATCH)
                    max_aspect = max(e1["aspect_ratio"], e2["aspect_ratio"])
                    if max_aspect > 0:
                        if abs(e1["aspect_ratio"] - e2["aspect_ratio"]) / max_aspect > 0.15:
                            continue

                    match_found = True
                    matched_v1_indices.add(i1)
                    break
                    
                if not match_found:
                    candidates.append(e2)

        # Step 5: Cluster the Candidates
        added_elements = []
        
        if len(candidates) > 0:
            cents = np.array([e["centroid"] for e in candidates])
            if len(cents) >= 3:
                db = DBSCAN(eps=30.0, min_samples=3).fit(cents)
                labels = db.labels_
                
                unique_labels = set(labels)
                for l in unique_labels:
                    idxs = np.where(labels == l)[0]
                    cluster = [candidates[i] for i in idxs]
                    
                    b_boxes = np.array([e["bbox"] for e in cluster])
                    c_bbox = [
                        float(np.min(b_boxes[:, 0])), float(np.min(b_boxes[:, 1])),
                        float(np.max(b_boxes[:, 2])), float(np.max(b_boxes[:, 3]))
                    ]
                    c_area = (c_bbox[2] - c_bbox[0]) * (c_bbox[3] - c_bbox[1])
                    
                    if l == -1:
                        # Singletons check against min_cluster_area
                        for e in cluster:
                            if e["area"] >= min_cluster_area:
                                added_elements.append({
                                    "bbox": e["bbox"],
                                    "label": "[+] ADDED",
                                    "primitives": 1
                                })
                    else:
                        if c_area >= min_cluster_area:
                            added_elements.append({
                                "bbox": c_bbox,
                                "label": "[+] ADDED",
                                "primitives": len(cluster)
                            })
            else:
                # Less than 3 elements total, only pass singletons if >= min_cluster_area
                for e in candidates:
                    if e["area"] >= min_cluster_area:
                        added_elements.append({
                            "bbox": e["bbox"],
                            "label": "[+] ADDED",
                            "primitives": 1
                        })
                        
        doc1.close()
        doc2.close()
        print(f"Extraction successful: Identified {len(added_elements)} structural ADDED bundles.")
        return added_elements
        
    def generate_3_panel(self, path_v1, path_v2, added_elements, output_path):
        def render_pdf_to_cv2(path):
            doc = fitz.open(path)
            page = doc[0]
            pix = page.get_pixmap(dpi=220)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            # Ensure BGR
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
            w, h = page.rect.width, page.rect.height
            doc.close()
            return img, w, h
            
        img1, pdf_w1, pdf_h1 = render_pdf_to_cv2(path_v1)
        img2, pdf_w2, pdf_h2 = render_pdf_to_cv2(path_v2)
        diff_img = img2.copy()
        
        # Scaling factor from PDF points to Pixels
        scale_x = img2.shape[1] / pdf_w2
        scale_y = img2.shape[0] / pdf_h2
        
        for added in added_elements:
            bbox = added["bbox"]
            # Convert PDF points to raster coordinates
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            
            cv2.rectangle(diff_img, (x1, y1), (x2, y2), (0, 200, 0), 4) # Green
            cv2.putText(diff_img, f"{added['label']} (n={added['primitives']})", (x1, max(y1-15, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

        # Build Headers
        def add_header(img, title):
            h, w = img.shape[:2]
            header = np.full((120, w, 3), 30, dtype=np.uint8)
            cv2.putText(header, title, (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            return np.vstack([header, img])

        p1 = add_header(img1, "V1 / ORIGINAL")
        p2 = add_header(img2, "V2 / REVISION")
        p3 = add_header(diff_img, "STAGE 1: ADDED ANALYSIS")

        if p1.shape != p2.shape:
             p1 = cv2.resize(p1, (p2.shape[1], p2.shape[0]))
             
        composite = np.hstack([p1, p2, p3])
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, composite)
        print(f"Isolated Stage 1 Result written to: {output_path}")

if __name__ == '__main__':
    detector = Stage1AddedDetector()
    
    for doc_num in range(138, 148):
        v1 = f"Drawings/PRV73B124{doc_num}.PDF"
        v2 = f"Drawings/PRV73B124{doc_num} - Copy.PDF"
        out_img = f"visuals_test/stage1_{doc_num}.jpg"
        
        if os.path.exists(v1) and os.path.exists(v2):
            print(f"\\n--- Processing 1241{doc_num} ---")
            added_list = detector.detect_added(v1, v2)
            detector.generate_3_panel(v1, v2, added_list, out_img)
