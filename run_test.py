import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import sys, time, os, cv2, numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, r'C:\Trivim Internship\engineering_comparison_system')

from comparator import compare
from report_generator import ReportGenerator
from pdf_reader import pdf_to_image

drawings = r'C:\Trivim Internship\engineering_comparison_system\Drawings'
REPORT_DIR = Path(r'C:\Trivim Internship\engineering_comparison_system\reports')

pairs = [
    ('PRV73B124138 - Copy.PDF',  'PRV73B124138.PDF'), # V2 (Original) has the dimensions
    ('PRV73B124139.PDF',        'PRV73B124139 - Copy.PDF'),
    ('PRV73B124140.PDF',        'PRV73B124140 - Copy.PDF'),
    ('PRV73B124141.PDF',        'PRV73B124141 - Copy.PDF'),
    ('PRV73B124142.PDF',        'PRV73B124142 - Copy.PDF'),
    ('PRV73B124143.PDF',        'PRV73B124143 - Copy.PDF'),
    ('PRV73B124144.PDF',        'PRV73B124144 - Copy.PDF'),
    ('PRV73B124145.PDF',        'PRV73B124145 - Copy.PDF'),
    ('PRV73B124146.PDF',        'PRV73B124146 - Copy.PDF'),
    ('PRV73B124147.PDF',        'PRV73B124147 - Copy.PDF'),
    ('PRQ93B101928 (1).pdf',     'PRQ93B101928_ori (1).pdf'),
]

def run_test():
    reporter = ReportGenerator()
    summary_data = []
    
    print(f"\n{'='*70}\nENGINEERING DRAWING COMPARISON TEST SUITE\n{'='*70}")
    
    for v1_name, v2_name in pairs:
        drawing_id = os.path.splitext(v1_name)[0]
        print(f'\n- Processing: {drawing_id}')
        
        v1_path = os.path.join(drawings, v1_name)
        v2_path = os.path.join(drawings, v2_name)
        
        t_start = time.time()
        try:
            # 1. Run Comparison Engine
            result = compare(v1_path, v2_path, drawing_id=drawing_id)
            elapsed = time.time() - t_start
            
            # 2. Map Results for Report Generator (Using new cluster schema)
            if result.verdict == "RASTER_REJECTED":
                change_res = {
                    "verdict": "RASTER_REJECTED", "added_count": 0, "removed_count": 0,
                    "modified_count": 0, "matched_count": 0
                }
            else:
                change_res = {
                    "verdict": result.verdict,
                    "added_count": len(result.geometry.get("added", [])),
                    "removed_count": len(result.geometry.get("removed", [])),
                    "modified_count": len(result.geometry.get("resized", [])),
                    "matched_count": 0
                }
            
            print(f'   Success: {result.verdict}')
            
            summary_data.append({
                "id": drawing_id,
                "verdict": result.verdict,
                "added": change_res["added_count"],
                "removed": change_res["removed_count"],
                "report": f"reports/report_{drawing_id}.xlsx"
            })

        except Exception as e:
            print(f'   ERROR: {e}')
            import traceback; traceback.print_exc()

    # 5. GENERATE MASTER SUMMARY FILE
    master_summary_path = REPORT_DIR / "latest_run_summary.md"
    with open(master_summary_path, "w", encoding="utf-8") as f:
        f.write("# Engineering Comparison System - Master Summary\n")
        f.write(f"**Run Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| Drawing ID | Verdict | Added | Removed | Full Report |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for s in summary_data:
            report_link = f"[View XLSX]({s['report']})" if s['report'] else "N/A"
            f.write(f"| **{s['id']}** | {s['verdict']} | {s['added']} | {s['removed']} | {report_link} |\n")
    
    print(f"\n{'='*70}\nMASTER SUMMARY GENERATED: {master_summary_path}\n{'='*70}")

if __name__ == "__main__":
    run_test()
