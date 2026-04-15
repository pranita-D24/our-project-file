# hf_audit.py
import os
import sys
import time
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Fix encoding for Windows terminals
if sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HF_Audit")

from comparator import compare
from report_generator import ReportGenerator

def run_hf_audit():
    print("=== STARTING HIGH-FIDELITY AUDIT (FIRST 10 PAIRS) ===")
    drawings_dir = Path("Drawings")
    results = []
    
    # Identify first 10 pairs
    originals = sorted(list(drawings_dir.glob("PRV73B*.PDF")))
    originals = [f for f in originals if "- Copy" not in f.name][:10]
    
    print(f"Targeting {len(originals)} pairs for audit.\n")
    
    rep_gen = ReportGenerator()
    
    for i, orig in enumerate(originals, 1):
        copy_name = orig.stem + " - Copy.PDF"
        copy_path = drawings_dir / copy_name
        
        if not copy_path.exists():
            continue
            
        print(f"\nProcessing {orig.name} vs {copy_name}...")
        start = time.time()
        
        try:
            res = compare(str(orig), str(copy_path))
            elapsed = time.time() - start
            
            # --- Rasterize images for report visuals ---
            from pdf_reader import pdf_to_image
            import cv2
            def get_gray(p):
                img = pdf_to_image(str(p), dpi=220)
                if img is None: return None
                img = cv2.resize(img, (1920, 1440))
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return g if np.mean(g) > 80 else cv2.bitwise_not(g)

            og_gray = get_gray(orig)
            mg_gray = get_gray(copy_path)

            # Generate Report
            reporter = ReportGenerator()
            
            # Prepare data for report_generator
            def region_to_obj(r):
                d = r.__dict__.copy()
                d["bbox"] = (r.x, r.y, r.w, r.h)
                return d

            # Support stripped comparator output
            g = res.geometry if hasattr(res, 'geometry') and res.geometry else {}
            match_res = {
                "added": g.get("added", []),
                "removed": g.get("removed", []),
                "moved": g.get("moved", []),
                "resized": g.get("resized", [])
            }
            change_res = {
                "similarity": res.similarity,
                "verdict": res.verdict,
                "added_count": len(match_res["added"]),
                "removed_count": len(match_res["removed"]),
                "modified_count": len(match_res["resized"]),
                "matched_count": res.processing_info.get("components_v1", 0) - len(match_res["removed"])
            }
            
            pdf_path = reporter.generate_enterprise_pdf(
                orig_gray=og_gray,
                mod_gray=mg_gray,
                change_result=change_res,
                match_result=match_res,
                dim_result={"line_length_changes": [], "changed_dims": []},
                ai_result=None,
                drawing_id=orig.stem,
                version_1="V1",
                version_2="V2",
                comparison_id=f"HF_AUDIT_{int(time.time())}"
            )
            
            total_changes_counted = change_res["added_count"] + change_res["removed_count"] + change_res["modified_count"] + len(match_res["moved"])
            results.append({
                "Pair": orig.stem,
                "Changes": total_changes_counted,
                "SSIM": f"{res.overall_ssim:.4f}",
                "YOLO_Fallback": "YES" if res.processing_info.get("yolo_fallback") else "NO",
                "Time": f"{elapsed:.2f}s",
                "PDF": str(pdf_path)
            })
            
            print(f"  Done. PDF: {pdf_path}")
            
        except Exception as e:
            print(f"  Error on {orig.stem}: {e}")
            # Ensure we still have an entry for the summary table
            results.append({
                "Pair": orig.stem,
                "Changes": "FAILED",
                "SSIM": "0.0000",
                "YOLO_Fallback": "N/A",
                "Time": "0s",
                "PDF": "#"
            })
            import traceback
            traceback.print_exc()

    # Final Summary Table (Console)
    print("\n\n" + "="*85)
    print(f"{'Pair':<20} | {'Changes Found':<14} | {'SSIM':<8} | {'YOLO Fallback':<14} | {'Time (s)'}")
    print("-" * 85)
    for r in results:
        # Use .get() defensively (User fix for KeyError)
        pair = str(r.get('Pair', 'N/A'))
        chg  = str(r.get('Changes', '0'))
        ssim = str(r.get('SSIM', '0.0000'))
        yolo = str(r.get('YOLO_Fallback', 'N/A'))
        t    = str(r.get('Time', '0s'))
        print(f"{pair:<20} | {chg:<14} | {ssim:<8} | {yolo:<14} | {t}")
    print("="*85)

    # WRITE TO MASTER REPORT FILE
    report_file = Path("reports/audit_summary.md")
    write_header = not report_file.exists()
    
    with open(report_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("# Engineering Drawing Comparison - Audit Summary\n\n")
            f.write("| Pair ID | Changes | SSIM | YOLO Fallback | Time (s) | Report Link |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for r in results:
            f.write(f"| **{r.get('Pair','N/A')}** | {r.get('Changes','0')} | {r.get('SSIM','0.0000')} | {r.get('YOLO_Fallback','N/A')} | {r.get('Time','0s')} | [Open PDF]({r.get('PDF','#')}) |\n")
            
    print(f"\nAudit Complete. Summary recorded in {str(report_file)}")

if __name__ == "__main__":
    run_hf_audit()
