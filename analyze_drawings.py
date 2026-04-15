# analyze_drawings.py
import os
import sys
import glob
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

import logging
logging.basicConfig(level=logging.INFO)

from comparator import compare

def run_batch_analysis():
    print("=== STARTING 10-SET DRAWING ANALYSIS ===")
    drawings_dir = Path("Drawings")
    results = []
    
    # Identify pairs
    all_pdfs = list(drawings_dir.glob("*.PDF"))
    originals = [f for f in all_pdfs if " - Copy" not in f.name]
    
    print(f"Found {len(originals)} potential pairs.\n")
    
    for orig in originals:
        copy_name = orig.stem + " - Copy.PDF"
        copy_path = drawings_dir / copy_name
        
        if not copy_path.exists():
            print(f"Warning: No copy found for {orig.name}")
            continue
            
        print(f"Analyzing Set: {orig.stem}")
        start_t = time.perf_counter()
        
        try:
            res = compare(str(orig), str(copy_path))
            elapsed = time.perf_counter() - start_t
            
            info = res.processing_info
            results.append({
                "ID": orig.stem,
                "Verdict": res.verdict,
                "Similarity": f"{res.similarity}%",
                "Added": info.get("added", 0),
                "Removed": info.get("removed", 0),
                "Moved": info.get("moved", 0),
                "DimChg": info.get("dim_changes", 0),
                "Time": f"{elapsed:.2f}s"
            })
            print(f"  Result: {res.verdict} ({res.similarity}%) in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"  Error processing {orig.stem}: {e}")
            results.append({
                "ID": orig.stem,
                "Verdict": "ERROR",
                "Similarity": "N/A",
                "Added": "N/A", "Removed": "N/A", "Moved": "N/A", "DimChg": "N/A",
                "Time": "N/A"
            })

    # Output Results Table
    print("\n\n" + "="*80)
    print(f"{'DRAWING_ID':<20} | {'VERDICT':<15} | {'SIMIL':<7} | {'ADD':<3} | {'REM':<3} | {'MOV':<3} | {'DIM':<3} | {'TIME'}")
    print("-" * 80)
    for r in results:
        print(f"{r['ID']:<20} | {r['Verdict']:<15} | {r['Similarity']:<7} | {r['Added']:<3} | {r['Removed']:<3} | {r['Moved']:<3} | {r['DimChg']:<3} | {r['Time']}")
    print("="*80)

if __name__ == "__main__":
    run_batch_analysis()
