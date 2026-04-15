# annotation_matcher.py — v2.1 Cost-Based Matching
import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any
from annotation_pipeline import Annotation

logger = logging.getLogger(__name__)

def compare_annotations(a1: List[Annotation], a2: List[Annotation], profile) -> Dict[str, List[Dict[str, Any]]]:
    """
    Matches two lists of annotations using the Hungarian Algorithm (cost-based assignment).
    Classifies results into SAME, DIM_CHANGE, ADDED, and REMOVED.
    """
    results = {
        "dim_changes": [],
        "added": [],
        "removed": [],
        "identical": []
    }
    
    if not a1 and not a2:
        return results
        
    if not a1:
        results["added"] = [vars(a) for a in a2]
        return results
        
    if not a2:
        results["removed"] = [vars(a) for a in a1]
        return results

    # ══════════════════════════════════════
    # BUILD COST MATRIX
    # ══════════════════════════════════════
    n, m = len(a1), len(a2)
    # Increase matrix size to handle unmatched items as high cost
    cost_matrix = np.full((n, m), 1e6) 
    
    move_thr = profile.move_threshold_px
    
    for i in range(n):
        for j in range(m):
            # Same type filter
            if a1[i].type != a2[j].type:
                continue
                
            # Centroid distance
            dist = np.hypot(a1[i].cx - a2[j].cx, a1[i].cy - a2[j].cy)
            if dist > move_thr:
                continue
                
            # Value similarity (GATING: gating check (10%))
            v1, v2 = a1[i].value, a2[j].value
            if v1 is not None and v2 is not None:
                if v1 != 0:
                    rel_diff = abs(v1 - v2) / abs(v1)
                else:
                    rel_diff = abs(v2) if v2 != 0 else 0
                
                if rel_diff > 0.10: # >10% is NOT a match
                    continue
                
                # Cost is dist + value penalty
                cost = dist + rel_diff * 100
            else:
                # One or both values missing — match by distance only
                cost = dist

            cost_matrix[i, j] = cost

    # ══════════════════════════════════════
    # HUNGARIAN ASSIGNMENT
    # ══════════════════════════════════════
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_a1 = set()
    matched_a2 = set()
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] > 5e5: # Threshold for "not assigned"
            continue
            
        matched_a1.add(i)
        matched_a2.add(j)
        
        obj1 = a1[i]
        obj2 = a2[j]
        
        # DIM_CHANGE classification (0.5% threshold)
        v1, v2 = obj1.value, obj2.value
        is_change = False
        if v1 is not None and v2 is not None:
            if v1 != 0:
                if abs(v1 - v2) / abs(v1) > 0.005:
                    is_change = True
            elif v2 != 0:
                is_change = True
        
        if is_change:
            results["dim_changes"].append({
                "v1": vars(obj1),
                "v2": vars(obj2),
                "type": "DIM_CHANGE",
                "detail": f"{obj1.text} -> {obj2.text}"
            })
        else:
            results["identical"].append(vars(obj2))

    # ADDED / REMOVED
    for i in range(n):
        if i not in matched_a1:
            results["removed"].append(vars(a1[i]))
            
    for j in range(m):
        if j not in matched_a2:
            results["added"].append(vars(a2[j]))

    return results
