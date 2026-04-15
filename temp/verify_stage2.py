# verify_stage2.py — Stage 2 Verification Suite
import sys, os, io
sys.path.insert(0, os.path.abspath('.'))

# Wrap stdout for UTF-8 handling on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import cv2
from annotation_pipeline import normalize_dim_value, extract_rotated_patch, Annotation, run_annotation_pipeline
from annotation_matcher import compare_annotations
from dataclasses import dataclass

@dataclass
class MockProfile:
    content_bbox: tuple = (0, 0, 5000, 5000)
    min_component_area: float = 100
    move_threshold_px: float = 50

def test_normalization():
    print("--- Testing Normalization ---")
    cases = [
        ("⌀50 ±0.1", (50.0, 0.1, "dimension")),
        ("45°", (45.0, None, "angle")),
        ("R10", (10.0, None, "radius")),
        ("100,5", (100.5, None, "dimension")),
        ("±0.05", (None, 0.05, "dimension")),
        (".25", (0.25, None, "dimension")),
        ("M10x1.5", (10.0, None, "thread")),
        ("2X ⌀15", (15.0, None, "dimension")),
    ]
    
    for text, expected in cases:
        val, tol, atype = normalize_dim_value(text)
        passed = (val == expected[0]) and (tol == expected[1]) and (atype == expected[2])
        print(f"INPUT: {text:10} | RESULT: {val}, {tol}, {atype:10} | PASS: {passed}")

def test_matching():
    print("\n--- Testing Hungarian Matching ---")
    profile = MockProfile()
    
    # v1: two annotations
    a1 = [
        Annotation("1", "dimension", (0,0,10,10), 100, 100, 10, 10, 0, "10", 10.0, None, 1.0),
        Annotation("2", "dimension", (0,0,10,10), 200, 200, 10, 10, 0, "20", 20.0, None, 1.0),
    ]
    
    # v2: shifted and changed
    a2 = [
        Annotation("3", "dimension", (0,0,10,10), 105, 105, 10, 10, 0, "10.1", 10.1, None, 1.0), # Matched, no change (<0.5%)
        Annotation("4", "dimension", (0,0,10,10), 205, 205, 10, 10, 0, "25", 25.0, None, 1.0),   # Matched, changed (>0.5%)
        Annotation("5", "dimension", (0,0,10,10), 400, 400, 10, 10, 0, "40", 40.0, None, 1.0),   # Added
    ]
    
    results = compare_annotations(a1, a2, profile)
    
    print(f"DIM_CHANGES: {len(results['dim_changes'])} (Expected: 1)")
    print(f"ADDED:       {len(results['added'])} (Expected: 1)")
    print(f"REMOVED:     {len(results['removed'])} (Expected: 0)")
    print(f"IDENTICAL:   {len(results['identical'])} (Expected: 1)")
    
    passed = len(results['dim_changes']) == 1 and len(results['added']) == 1
    print(f"PASS: {passed}")

def test_pipeline_stub():
    print("\n--- Testing Pipeline Stub ---")
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    mask = np.zeros((1000, 1000), dtype=np.uint8)
    profile = MockProfile()
    
    anns = run_annotation_pipeline(img, profile, mask)
    print(f"Annotations found: {len(anns)}")
    if len(anns) > 0:
        print(f"First result: {anns[0].type} at ({anns[0].cx}, {anns[0].cy}) val={anns[0].value}")
    
    passed = len(anns) > 0
    print(f"PASS: {passed}")

if __name__ == "__main__":
    test_normalization()
    test_matching()
    test_pipeline_stub()
