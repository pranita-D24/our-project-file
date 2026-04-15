import numpy as np
import logging
from typing import List, Dict, Any, Tuple
import fitz
import re
from sklearn.ensemble import IsolationForest

def get_centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def detect_balloons(page: fitz.Page, text_dict: Dict[str, Any], drawing_id: str, drawings: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Advanced Balloon Detection using Isolation Forest.
    Returns:
        {
            "balloons_ignored": int,
            "path_indices": Set[int],
            "text_span_indices": Set[Tuple[int, int, int]],
            "locations": List[Tuple[float, float]]
        }
    """
    if drawings is None:
        drawings = page.get_drawings()
    page_rect = page.rect
    
    # Traceability
    logger = logging.getLogger("Stage3")
    
    # 1. Step: Extract curve-only paths
    candidate_indices = []
    for i, p in enumerate(drawings):
        if all(item[0] in ['c', 'qu'] for item in p['items']):
            candidate_indices.append(i)
            
    if len(candidate_indices) < 4:
        # logging.info(f"[STAGE3] [{drawing_id}] Fewer than 4 curve paths. Skipping ML.")
        return {"balloons_ignored": 0, "path_indices": set(), "text_span_indices": set(), "locations": []}

    # 2. Step: Concentricity Pre-Filter
    # Bore holes share centroids. Balloons are usually solitary.
    concentric_indices = set()
    centroids = [get_centroid(drawings[idx]['rect']) for idx in candidate_indices]
    
    for i in range(len(candidate_indices)):
        for j in range(i + 1, len(candidate_indices)):
            c1, c2 = centroids[i], centroids[j]
            dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
            if dist < 1.0: # Concentric!
                concentric_indices.add(candidate_indices[i])
                concentric_indices.add(candidate_indices[j])

    ml_candidate_indices = [idx for idx in candidate_indices if idx not in concentric_indices]
    
    if len(ml_candidate_indices) < 4:
        return {"balloons_ignored": 0, "path_indices": set(), "text_span_indices": set(), "locations": []}

    # 3. Step: Feature Matrix for Isolation Forest
    # Features: [Area, Aspect Ratio, X_norm, Y_norm]
    features = []
    for idx in ml_candidate_indices:
        r = drawings[idx]['rect']
        w, h = r.width, r.height
        area = w * h
        aspect = w / h if h != 0 else 0
        c = get_centroid(r)
        features.append([
            area,
            aspect,
            c[0] / page_rect.width,
            c[1] / page_rect.height
        ])
        
    features = np.array(features)
    
    # 4. Step: Run Isolation Forest
    clf = IsolationForest(contamination='auto', random_state=42)
    labels = clf.fit_predict(features) # -1 = Anomaly
    
    anomaly_indices = [ml_candidate_indices[i] for i in range(len(labels)) if labels[i] == -1]

    # 5. Step: Post-Filter Text Containment
    # An anomaly must contain a 1-3 digit numeric text to be a balloon
    balloons_found = []
    text_indices = set()
    path_indices = set()
    locations = []
    
    for p_idx in anomaly_indices:
        r = drawings[p_idx]['rect']
        found_text = False
        for b_idx, block in enumerate(text_dict.get("blocks", [])):
            if "lines" not in block: continue
            for l_idx, line in enumerate(block["lines"]):
                for s_idx, span in enumerate(line["spans"]):
                    txt = span["text"].strip()
                    if re.match(r'^\d{1,3}$', txt):
                        tc = get_centroid(span["bbox"])
                        # Check inside circle bbox expanded by 5pts
                        if (r[0]-5 <= tc[0] <= r[2]+5 and r[1]-5 <= tc[1] <= r[3]+5):
                            text_indices.add((b_idx, l_idx, s_idx))
                            path_indices.add(p_idx)
                            locations.append(get_centroid(r))
                            found_text = True
                            break
                if found_text: break
            if found_text: break
            
    return {
        "balloons_ignored": len(path_indices),
        "path_indices": path_indices,
        "text_span_indices": text_indices,
        "locations": locations
    }
