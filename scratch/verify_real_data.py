import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from stage2_vector import Stage2Engine
import fitz
import re

def run_real_verification():
    engine = Stage2Engine()
    
    # Paths corrected: V1 is the Copy (Old), V2 is the main file (New)
    v1 = r"c:\Trivim Internship\engineering_comparison_system\drawings\PRV73B124138 - Copy.PDF"
    v2 = r"c:\Trivim Internship\engineering_comparison_system\drawings\PRV73B124138.PDF"
    
    output_dir = r"c:\Trivim Internship\engineering_comparison_system\output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("VERIFYING STAGES 2+3 ON REAL DRAWING PRV73B124138")
    print("==================================================")
    
    doc1 = fitz.open(v1)
    doc2 = fitz.open(v2)
    
    d1 = engine.extract_page_data(doc1[0], "PRV73B124138-V1")
    d2 = engine.extract_page_data(doc2[0], "PRV73B124138-V2")
    
    # Match
    m, rem, add = engine.match_entities(d1["dimensions"], d2["dimensions"], 15.0)
    
    print(f"\nBalloons Detected (V2): {d2.get('balloons_count', 0)}")
    
    added_values = [e.value for e in add]
    targets = ["60", "89.650.05"]
    
    print("\nChecking Target Dimensions Added Status:")
    for target in targets:
        match = [v for v in added_values if target in v]
        print(f"  Target '{target}' in ADDED? {'YES' if match else 'NO'}")
    
    print("\nAll Verified Dimensions in V2:")
    for d in d2["dimensions"]:
        print(f"  - {d.value}")
        
    doc1.close()
    doc2.close()

if __name__ == "__main__":
    run_real_verification()
