# verify_stage5_6.py
import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from report_generator import ReportGenerator

def test_enterprise_pdf():
    print("--- Testing Stage 5: Enterprise PDF Report ---")
    rg = ReportGenerator()
    
    # Mock data
    h, w = 600, 800
    orig = np.zeros((h, w), dtype=np.uint8)
    mod  = np.zeros((h, w), dtype=np.uint8)
    cv2_result = {
        "verdict": "MODERATELY DIFFERENT",
        "similarity": 85,
        "added_count": 2,
        "removed_count": 1,
        "modified_count": 3,
        "matched_count": 10
    }
    match_result = {
        "added": [{"bbox": [10, 10, 50, 50], "type": "DIM"}],
        "removed": [],
        "modified": []
    }
    
    pdf_path = rg.generate_enterprise_pdf(
        orig, mod, cv2_result, match_result, 
        None, None, "TEST-DWG-001", "v1", "v2", "X-999"
    )
    
    print(f"PDF Generated: {pdf_path}")
    if pdf_path and os.path.exists(pdf_path):
        print("PDF Verification: PASSED")
        # Check size > 0
        if os.path.getsize(pdf_path) > 0:
            print(f"PDF Size: {os.path.getsize(pdf_path)} bytes")
    else:
        print("PDF Verification: FAILED")

def test_scaling_stubs():
    print("\n--- Testing Stage 6: Scaling Stubs ---")
    files = ["celery_worker.py", "dag_orchestrator.py"]
    for f in files:
        exists = os.path.exists(f)
        print(f"File {f} exists: {exists}")
        if not exists:
            print(f"ERROR: {f} is missing.")

if __name__ == "__main__":
    test_enterprise_pdf()
    test_scaling_stubs()
