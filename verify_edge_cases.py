# verify_edge_cases.py
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import time
import cv2
from comparator import _match_components, _shape_score, _normalize_text

def get_mock_element(id, px, py, text="10", shape="rectangle", w=10, h=10):
    return {
        "id": id,
        "centroid": (px + w/2.0, py + h/2.0),
        "area": w * h,
        "bbox": (px, py, w, h),
        "text": text,
        "hu": [1.0] * 7,
        "sig": [1.0] * 32,
        "cnt": np.array([[[0,0]], [[1,1]], [[2,2]]], dtype=np.int32),
        "shape": shape,
        "confidence": 0.95
    }

def run_tests():
    print("=== ENGINEERING STRESS TEST (STAGE 4) ===")
    g = np.zeros((500, 500), dtype=np.uint8)
    
    # CASE A: Type Coercion (int 10 vs str "10.0")
    print("\n[STRESS A] Type Coercion: 10 (int) vs '10.0' (str)")
    c1 = [get_mock_element("1", 100, 100, text=10)]
    c2 = [get_mock_element("2", 100, 100, text="10.0")]
    matched, _, _ = _match_components(c1, c2, g, g)
    label = matched[0][0] if matched else "NONE"
    print(f"RESULT: {label} (Expected: MATCH)")

    # CASE B: Tolerance Precision (±0.025 vs ±0.0250)
    print("\n[STRESS B] Tolerance Precision: ±0.025 vs ±0.0250")
    c1 = [get_mock_element("1", 100, 100, text="±0.025")]
    c2 = [get_mock_element("2", 100, 100, text="±0.0250")]
    matched, _, _ = _match_components(c1, c2, g, g)
    label = matched[0][0] if matched else "NONE"
    print(f"RESULT: {label} (Expected: MATCH)")
    
    # CASE C: Symbol Prefix (⌀50 vs 50)
    print("\n[STRESS C] Symbol Prefix: ⌀50 vs 50")
    c1 = [get_mock_element("1", 100, 100, text="⌀50")]
    c2 = [get_mock_element("2", 100, 100, text="50")]
    matched, _, _ = _match_components(c1, c2, g, g)
    label = matched[0][0] if matched else "NONE"
    print(f"RESULT: {label} (Expected: MATCH)")

    # CASE D: Rotation Invariance (10x20 at same point)
    print("\n[STRESS D] Rotation Invariance (10x20 vs 20x10)")
    c1_rot = get_mock_element("1", 100, 100, w=10, h=20)
    c2_rot = get_mock_element("2", 100, 100, w=20, h=10)
    matched, _, _ = _match_components([c1_rot], [c2_rot], g, g)
    label = matched[0][0] if matched else "NONE"
    print(f"RESULT: {label} (Expected: MATCH)")

    # CASE E: Precision Guard (±0.0025 vs ±0.0026)
    print("\n[STRESS E] Precision Guard: ±0.0025 vs ±0.0026")
    c1_p = get_mock_element("1", 100, 100, text="±0.0025")
    c2_p = get_mock_element("2", 100, 100, text="±0.0026")
    matched, _, _ = _match_components([c1_p], [c2_p], g, g)
    label = matched[0][0] if matched else "NONE"
    print(f"RESULT: {label} (Expected: DIM_CHANGED)")

if __name__ == "__main__":
    run_tests()
