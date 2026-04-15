from sklearn.cluster import DBSCAN
import re
import logging
import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("Stage5")

def get_drawing_regions(page_text_dict: dict) -> List[Dict[str, Any]]:
    """
    Scans for View Labels (SECTION, DETAIL, VIEW, SCALE) to define Voronoi regions.
    """
    labels = []
    keywords = ["SECTION", "DETAIL", "VIEW", "SCALE"]
    
    for block in page_text_dict.get("blocks", []):
        if "lines" not in block: continue
        for line in block["lines"]:
            for span in line["spans"]:
                txt = span["text"].strip().upper()
                if any(k in txt for k in keywords):
                    # Store Label, Centroid
                    b = span["bbox"]
                    labels.append({
                        "name": span["text"].strip(),
                        "centroid": [(b[0]+b[2])/2, (b[1]+b[3])/2]
                    })
    return labels

def assign_to_region(centroid: Tuple[float, float], regions: List[Dict[str, Any]]) -> str:
    """
    Voronoi Assignment: Assigns the centroid to the nearest view label region.
    """
    if not regions: return "MAIN VIEW"
    
    min_dist = float('inf')
    best_region = "MAIN VIEW"
    
    cx, cy = centroid
    for r in regions:
        rx, ry = r["centroid"]
        dist = ((cx - rx)**2 + (cy - ry)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            best_region = r["name"]
            
    return best_region

def discover_moves(added: List[dict], removed: List[dict], page_rect: Any, page_text_dict: dict) -> Dict[str, Any]:
    """
    Identifies moved components between revisions.
    Supports 0, 90, 180 degree rotations.
    Uses Voronoi partitioning for region detection.
    """
    moved = []
    final_added = []
    
    # Page Diagonal for displacement thresholding (User: < 30% diagonal)
    diag = (page_rect.width**2 + page_rect.height**2)**0.5
    move_threshold = diag * 0.3
    
    # 1. Build signature index for REMOVED items
    # Key: (type, op_signature, width, height)
    # 90-degree check requires checking swapped W/H
    removed_pool = {}
    for r in removed:
        bbox = r["bbox"]
        w, h = round(bbox[2]-bbox[0], 1), round(bbox[3]-bbox[1], 1)
        key = (r["type"], r["op_signature"], w, h)
        removed_pool.setdefault(key, []).append(r)

    regions = get_drawing_regions(page_text_dict)
    
    # 2. Track which removed items are consumed
    used_removed_hashes = set()

    for a in added:
        bbox_a = a["bbox"]
        wa, ha = round(bbox_a[2]-bbox_a[0], 1), round(bbox_a[3]-bbox_a[1], 1)
        
        match_found = False
        # Potential keys: 0/180 deg (wa, ha) or 90/270 deg (ha, wa)
        potential_keys = [
            (a["type"], a["op_signature"], wa, ha), # 0, 180
            (a["type"], a["op_signature"], ha, wa)  # 90
        ]
        
        for p_key in potential_keys:
            if p_key in removed_pool:
                for r in removed_pool[p_key]:
                    r_id = id(r)
                    if r_id in used_removed_hashes: continue
                    
                    # Distance check
                    dx = a["centroid"][0] - r["centroid"][0]
                    dy = a["centroid"][1] - r["centroid"][1]
                    dist = (dx**2 + dy**2)**0.5
                    
                    if dist < move_threshold:
                        # Identify rotation
                        rotation = 0
                        if p_key[2] == ha and p_key[3] == wa and wa != ha:
                            rotation = 90
                        
                        moved.append({
                            "type": a["type"],
                            "status": "MOVED",
                            "signature": a["op_signature"],
                            "from_centroid": r["centroid"],
                            "to_centroid": a["centroid"],
                            "from_bbox": r["bbox"],
                            "to_bbox": a["bbox"],
                            "displacement_vector": [dx, dy],
                            "displacement_distance": dist,
                            "rotation": rotation,
                            "region": assign_to_region(a["centroid"], regions)
                        })
                        used_removed_hashes.add(r_id)
                        match_found = True
                        break
            if match_found: break
            
        if not match_found:
            a_copy = a.copy()
            a_copy["region"] = assign_to_region(a["centroid"], regions)
            a_copy["status"] = "ADDED"
            final_added.append(a_copy)

    # 3. Finalize REMOVED list
    final_removed = []
    for r in removed:
        if id(r) not in used_removed_hashes:
            r_copy = r.copy()
            r_copy["region"] = assign_to_region(r["centroid"], regions)
            r_copy["status"] = "REMOVED"
            final_removed.append(r_copy)
            
    return {
        "added": final_added,
        "removed": final_removed,
        "moved": moved
    }

def cluster_to_components(entities: List[dict], status: str) -> List[dict]:
    """
    Groups primitives into COMPONENTS using DBSCAN.
    Rule 1: 10+ items -> 1 Component Row.
    Rule 2: 2-9 items -> Individual rows with cluster_id.
    Rule 3: Isolated -> Individual row.
    """
    if not entities: return []
    
    centroids = np.array([e["centroid"] for e in entities])
    if len(centroids) < 2: 
        for e in entities: e["status"] = status
        return entities
        
    # Dynamic EPS based on 90th percentile of local NN distances
    tree = KDTree(centroids)
    nn_dists, _ = tree.query(centroids, k=2)
    eps = float(np.percentile(nn_dists[:, 1], 90))
    eps = max(min(eps, 40.0), 5.0) # Sane range for mechanical proximity
    
    # FIX 3: Strict 35% Epsilon Reduction
    # Reduces the radius to prevent stray noise from bridging legitimate clusters
    eps = eps * 0.65
    
    db = DBSCAN(eps=eps, min_samples=2).fit(centroids)
    labels = db.labels_
    
    results = []
    unique_labels = sorted(set(labels))
    
    for l in unique_labels:
        indices = np.where(labels == l)[0]
        cluster_items = [entities[i] for i in indices]
        
        b_boxes = np.array([item["bbox"] for item in cluster_items])
        bbox = [
            float(np.min(b_boxes[:, 0])), float(np.min(b_boxes[:, 1])),
            float(np.max(b_boxes[:, 2])), float(np.max(b_boxes[:, 3]))
        ]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # User defined heuristic: Must have min_area = 500 AND min_primitives = 3
        # Any cluster failing this check is considered admin noise/annotation artifact
        if area < 500.0 or len(cluster_items) < 3:
            continue
            
        if l == -1 or len(cluster_items) < 3:
            # Isolated / small groups (Normally this block wouldn't be hit with len<3 stripped above)
            for item in cluster_items:
                item["status"] = status
                item["component_type"] = "isolated"
                item["primitive_count"] = 1
                results.append(item)
        elif len(cluster_items) >= 10:
            # Rule 1: Collapse to Component
            # Determine dominant type
            types = [item["type"] for item in cluster_items]
            dom_type = max(set(types), key=types.count)
            
            # Aggregate stats
            c_centroids = np.array([item["centroid"] for item in cluster_items])
            center = np.mean(c_centroids, axis=0).tolist()
            
            results.append({
                "type": f"{dom_type}-cluster",
                "status": f"COMPONENT {status}",
                "component_type": dom_type,
                "primitive_count": len(cluster_items),
                "centroid": center,
                "bbox": bbox,
                "notes": f"System identified {len(cluster_items)} grouped primitives",
                "op_signature": cluster_items[0].get("op_signature", "complex")
            })
        else:
            # Rule 2: 3-9 items, keep individual but tag with cluster_id
            cluster_id = f"group_{status.lower()}_{l}"
            for item in cluster_items:
                item["status"] = status
                item["cluster_id"] = cluster_id
                item["component_type"] = item["type"]
                item["primitive_count"] = 1 # Individually reported
                results.append(item)
                
    return results
