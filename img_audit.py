# img_audit.py - Direct Image Audit Runner
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from comparator import compare
from report_generator import ReportGenerator

def run_img_audit():
    print("=== STARTING IMAGE-DIRECT PRECISION AUDIT (PRQ93B SERIES) ===")
    
    # Literal ingestion using verified absolute paths
    all_files = [
        Path(r"C:\Trivim Internship\drawing files\PRQ93B101905_17_V1.png"),
        Path(r"C:\Trivim Internship\drawing files\PRQ93B101905_17_V2.png"),
        Path(r"C:\Trivim Internship\drawing files\PRQ93B101928_00_V1.png"),
        Path(r"C:\Trivim Internship\drawing files\PRQ93B101928_00_V2.png")
    ]
    
    print(f"Ingested {len(all_files)} raw assets: {[f.name for f in all_files]}")
    
    # Pair them up by base ID (Forensic match)
    v1_files = sorted([f for f in all_files if "_V1" in f.name])
    pairs = []
    
    for v1 in v1_files:
        v2_target_name = v1.name.replace("_V1", "_V2")
        print(f"Targeting: {v2_target_name}")
        
        v2 = None
        for f in all_files:
            if f.name == v2_target_name:
                v2 = f
                break
        
        if v2:
            pairs.append({
                "id": v1.name.replace("_V1", "").replace(".png", ""),
                "v1": str(v1),
                "v2": str(v2)
            })
    
    print(f"Paired {len(pairs)} sets: {[p['id'] for p in pairs]}")
    
    results = []
    reporter = ReportGenerator()
    
    for p in pairs:
        print(f"\nProcessing {p['id']}...")
        print(f"  V1: {p['v1']}")
        print(f"  V2: {p['v2']}")
            
        start = time.time()
        try:
            # 1. Run Engine (Force Ingestion)
            res = compare(p['v1'], p['v2'])
            elapsed = time.time() - start
            print(f"  Similarity Match: {res.similarity:.4f}")
            
            # 2. Prepare Grayscale for Report
            def load_gray(path):
                img = cv2.imread(path)
                if img is None: return None
                img = cv2.resize(img, (1920, 1440))
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return g if np.mean(g) > 127 else cv2.bitwise_not(g)
            
            g1 = load_gray(p['v1'])
            g2 = load_gray(p['v2'])
            
            # 3. Generate Report
            def region_to_obj(r):
                d = r.__dict__.copy()
                d["bbox"] = (r.x, r.y, r.w, r.h)
                return d

            match_res = {
                "added": [region_to_obj(r) for r in res.regions if r.change_type == "ADDED"],
                "removed": [region_to_obj(r) for r in res.regions if r.change_type == "REMOVED"],
                "modified": [{"v1_object": region_to_obj(r), "v2_object": region_to_obj(r)} 
                             for r in res.regions if r.change_type == "RESIZED"]
            }
            # Prepare data for report_generator
            info = res.processing_info if res.processing_info else {}
            
            match_res = {
                "added": [region_to_obj(r) for r in res.regions if r.change_type == "ADDED"],
                "removed": [region_to_obj(r) for r in res.regions if r.change_type == "REMOVED"],
                "modified": [{"v1_object": region_to_obj(r), "v2_object": region_to_obj(r)} 
                             for r in res.regions if r.change_type == "RESIZED"]
            }
            change_res = {
                "similarity": res.similarity,
                "verdict": res.verdict,
                "added_count": len(match_res["added"]),
                "removed_count": len(match_res["removed"]),
                "modified_count": len(match_res["modified"]),
                "matched_count": info.get("components_v1", 0) - len(match_res["removed"])
            }
            
            # 3. Generate Full Report Suite
            report_paths = reporter.generate_all_reports(
                orig_gray=g1, mod_gray=g2,
                change_result=change_res,
                match_result=match_res,
                dim_result={"line_length_changes": [], "changed_dims": []},
                ai_result=None,
                drawing_id=p['id'],
                version_1="V1",
                version_2="V2",
                comparison_id=f"IMG_AUDIT_{int(time.time())}",
                processing_time=elapsed
            )
            
            pdf_path = report_paths.get("pdf")
            
            results.append({
                "Pair": p['id'],
                "Changes": len(res.regions),
                "SSIM": f"{res.overall_ssim:.4f}",
                "YOLO_Fallback": "N/A",
                "Time": f"{elapsed:.2f}s",
                "PDF": str(pdf_path)
            })
            print(f"  Done. PDF: {pdf_path}")
            
        except Exception as e:
            print(f"  Error on {p['id']}: {e}")
            import traceback; traceback.print_exc()

    # 4. Append to Summary Table
    summary_file = Path("reports/audit_summary.md")
    if summary_file.exists():
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write("\n### [NEW] Image-Based Audit (PRQ93B Series)\n")
            f.write("| Pair ID | Changes | SSIM | YOLO Fallback | Time (s) | Report Link |\n")
            for r in results:
                f.write(f"| **{r['Pair']}** | {r['Changes']} | {r['SSIM']} | {r['YOLO_Fallback']} | {r['Time']} | [Open PDF]({r['PDF']}) |\n")
        print(f"\nSuccess. Results appended to {summary_file}")

if __name__ == "__main__":
    run_img_audit()
