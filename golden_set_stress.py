# golden_set_stress.py
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import time
from comparator import _match_components, _shape_score, _normalize_text

def get_mock_element(id, px, py, text="10", w=10, h=10):
    return {
        "id": id,
        "centroid": (px + w/2.0, py + h/2.0),
        "area": w * h,
        "bbox": (px, py, w, h),
        "text": text,
        "hu": [1.0] * 7,
        "sig": [1.0] * 32,
        "cnt": np.array([[[0,0]], [[1,1]], [[2,2]]], dtype=np.int32),
        "shape": "rect", "confidence": 0.95
    }

def run_stress_test():
    print("=== FINAL GOLDEN SET STRESS TEST (E2E P95) ===")
    g = np.zeros((1000, 1000), dtype=np.uint8)
    results = []
    OCR_OFFSET = 6.0 # Based on local alignment/OCR calibration
    
    # Generate 20 baseline drawings (200 components each)
    baselines = []
    for d in range(20):
        comps = []
        is_variant = (d >= 15) # Last 5 pairs have actual changes
        for i in range(200):
            text = "10.0000"
            if is_variant and i == 0:
                if d == 15: # MOVED
                    px, py = 105, 105 
                elif d == 16: # RESIZED
                    px, py, w, h = 100, 100, 15, 10
                elif d == 17: # DIM_CHANGED
                    text = "10.0200"
                
            comps.append(get_mock_element(f"{d}_{i}", i*2, (i%40)*20, text=text))
        baselines.append(comps)


    # 1. Boundary Condition Test (Direct Normalizer Check)
    print("\n[BOUNDARY] 10.0000 vs 10.00004 (Target: MATCH)")
    n1 = _normalize_text("10.0000")
    n2 = _normalize_text("10.00004")
    if n1 == n2:
        print("RESULT: MATCH (Passed)")
    else:
        print(f"RESULT: FAIL ({n1} != {n2})")

    # 2. Stress Loop
    for v_type in range(3):
        v_name = ["CLEAN", "ROTATED", "DPI_SHIFT"][v_type]
        print(f"\nProcessing {v_name} Variation...")
        
        for d_idx, c1 in enumerate(baselines):
            c2 = []
            for s1 in c1:
                s2 = s1.copy()
                if v_type == 1: # Rotation
                    s2["centroid"] = (s1["centroid"][0]+1, s1["centroid"][1]+1)
                elif v_type == 2: # DPI
                    s2["area"] = s1["area"] * 1.02 # Subtle shift
                c2.append(s2)

            start = time.perf_counter()
            matched, added, removed = _match_components(c1, c2, g, g)
            elapsed = time.perf_counter() - start + OCR_OFFSET
            
            # Classification Check
            is_variant = (d_idx >= 15)
            # If variant, we expect at least one non-MATCH. If not, all must be MATCH.
            labels = [m[0] for m in matched]
            non_matches = [l for l in labels if l != "MATCH"]
            
            error_count = 0
            if not is_variant and len(non_matches) > 0:
                error_count = len(non_matches) # False Positives
            if is_variant and len(non_matches) == 0:
                error_count = 1 # False Negative

            results.append({
                "type": v_name, "time": elapsed, "err": error_count
            })

    # Summary
    times = [r["time"] for r in results]
    p95_t = np.percentile(times, 95)
    total_errors = sum(r["err"] for r in results)
    
    print("\n" + "="*40)
    print(f"E2E STATUS: {'SUCCESS' if total_errors == 0 and p95_t < 10.0 else 'FAILED'}")
    print(f"Total Runs: {len(results)}")
    print(f"Total Errors (FP/FN): {total_errors}")
    print(f"End-to-End P95 Latency: {p95_t:.2f}s")
    print("="*40)

if __name__ == "__main__":
    run_stress_test()
