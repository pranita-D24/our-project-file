import fitz
import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

# Add project root to path
sys.path.insert(0, r'C:\Trivim Internship\engineering_comparison_system')

from stage2_vector import Stage2Engine
from stage4_geometry import Stage4Engine
from stage3_balloons import detect_balloons

def debug_140():
    p1_path = r'Drawings\PRV73B124140.PDF'
    p2_path = r'Drawings\PRV73B124140 - Copy.PDF'
    
    doc1 = fitz.open(p1_path)
    doc2 = fitz.open(p2_path)
    page1 = doc1[0]
    page2 = doc2[0]
    
    s2 = Stage2Engine()
    s4 = Stage4Engine()
    
    b1 = detect_balloons(page1, page1.get_text("dict"), "V1")
    b2 = detect_balloons(page2, page2.get_text("dict"), "V2")
    
    d1 = s2.extract_page_data(page1, "V1")
    d2 = s2.extract_page_data(page2, "V2")
    
    bounds1 = s2.detect_boundaries(page1)
    bounds2 = s2.detect_boundaries(page2)
    
    print(f"Bounds V1: {bounds1}")
    
    v1_list = s4.extract_filtered(page1, b1["path_indices"], d1["dimensions"], bounds1)
    v2_list = s4.extract_filtered(page2, b2["path_indices"], d2["dimensions"], bounds2)
    
    print(f"Extracted V1 potential entities: {len(v1_list)}")
    print(f"Extracted V2 potential entities: {len(v2_list)}")
    
    # Analyze clusters in V1 potential removals
    if len(v1_list) > 0:
        centroids = np.array([e.centroid for e in v1_list])
        tree = KDTree(centroids)
        nn_dists, _ = tree.query(centroids, k=2)
        eps = float(np.percentile(nn_dists[:, 1], 90))
        eps = max(min(eps, 50.0), 2.0)
        
        db = DBSCAN(eps=eps, min_samples=5).fit(centroids)
        labels = db.labels_
        
        unique_labels = sorted(set(labels))
        print(f"\nCluster Analysis (eps={eps:.2f}):")
        for l in unique_labels:
            if l == -1: 
                print(f"  Noise points: {len(np.where(labels == -1)[0])}")
                continue
            indices = np.where(labels == l)[0]
            c_centroids = centroids[indices]
            c_tree = KDTree(c_centroids)
            c_nn, _ = c_tree.query(c_centroids, k=2)
            mean_nn, std_nn = np.mean(c_nn[:,1]), np.std(c_nn[:,1])
            cv = std_nn / mean_nn if mean_nn > 0 else 0
            print(f"  Cluster {l}: size={len(indices)}, cv={cv:.4f}, mean_dist={mean_nn:.2f}")

    res = s4.compare_pages(page1, page2, d1["dimensions"], d2["dimensions"], 
                           b1["path_indices"], b2["path_indices"], bounds1, bounds2)
    
    geo = res["geometry"]
    print(f"\nFinal Geometry Results:")
    print(f"  Added: {len(geo['added'])}")
    print(f"  Removed: {len(geo['removed'])}")
    print(f"  Resized: {len(geo['resized'])}")

if __name__ == "__main__":
    debug_140()
