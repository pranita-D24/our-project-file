from scipy.spatial import cKDTree
import numpy as np

def compare_spans(spans_v1: list, spans_v2: list, 
                  tolerance_px: float = 20.0) -> tuple:
    """
    Returns (added, removed) — spans that exist in one but not the other.
    Matches by BOTH spatial proximity AND text value.
    """
    if not spans_v1:
        return spans_v2, []
    if not spans_v2:
        return [], spans_v1

    # Build coordinate trees
    coords_v1 = np.array([s["centroid"] for s in spans_v1])
    coords_v2 = np.array([s["centroid"] for s in spans_v2])
    
    tree_v1 = cKDTree(coords_v1)

    matched_v1 = set()
    matched_v2 = set()

    # For each V2 span, find nearest V1 span within tolerance
    for i, s2 in enumerate(spans_v2):
        dist, idx = tree_v1.query(s2["centroid"], k=1)
        if dist <= tolerance_px:
            # Close enough spatially — check text matches
            if spans_v1[idx]["text"].strip() == s2["text"].strip():
                matched_v1.add(idx)
                matched_v2.add(i)

    # Unmatched = actual changes
    added   = [s for i,s in enumerate(spans_v2) if i not in matched_v2]
    removed = [s for i,s in enumerate(spans_v1) if i not in matched_v1]

    return added, removed
