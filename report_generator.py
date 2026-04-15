import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
import datetime
import logging
logger = logging.getLogger(__name__)

class ReportGenerator:
    def generate_all_reports(self, drawing_id, match_result, dim_result, change_result, version_1, version_2, **kwargs):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Audit Report"

        # Header
        ws.append(["Drawing ID", "Version From", "Version To", "Date", "Verdict"])
        ws.append([drawing_id, version_1, version_2, str(datetime.datetime.now().strftime("%Y-%m-%d")), change_result.get("verdict", "-")])
        ws.append([])

        # Only 4 counts
        ws.append(["Change Type", "Count"])
        ws.append(["ADDED",   len([x for x in match_result.get("added", [])   if x.get("status","").startswith("COMPONENT")])])
        ws.append(["REMOVED", len([x for x in match_result.get("removed", []) if x.get("status","").startswith("COMPONENT")])])
        ws.append(["MOVED",   len(match_result.get("moved", []))])
        ws.append(["RESIZED", len(match_result.get("resized", []))])

        import os
        out = f"Drawings/comparison_report_{drawing_id}.xlsx"
        wb.save(out)
        print(f"Report saved: {out}")

    def generate_master_summary(self, results, output_dir):
        pass

    def generate_enterprise_pdf(self, **kwargs):
        import cv2
        import numpy as np
        import os
        
        orig_gray = kwargs.get("orig_gray")
        drawing_id = kwargs.get("drawing_id", "Unknown")
        mod_gray = kwargs.get("mod_gray")
        match_result = kwargs.get("match_result", {})

        if mod_gray is None:
            logger.warning("No mod_gray provided to visual generator.")
            return None

        # Build 3 separate panels
        if len(mod_gray.shape) == 2:
            panel_v2 = cv2.cvtColor(mod_gray, cv2.COLOR_GRAY2BGR)
            canvas_diff = cv2.cvtColor(mod_gray, cv2.COLOR_GRAY2BGR)
        else:
            panel_v2 = mod_gray.copy()
            canvas_diff = mod_gray.copy()
            
        if orig_gray is not None:
            panel_v1 = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR) if len(orig_gray.shape) == 2 else orig_gray.copy()
            if panel_v1.shape != panel_v2.shape:
                panel_v1 = cv2.resize(panel_v1, (panel_v2.shape[1], panel_v2.shape[0]))
        else:
            panel_v1 = panel_v2.copy()

        def draw_boxes(items, color, label=""):
            for item in items:
                bbox = item.get("bbox", item.get("to_bbox", item.get("from_bbox")))
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(canvas_diff, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(canvas_diff, label, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Draw Clean Minimal Markers on the Diff Panel
        draw_boxes(match_result.get("added", []), (0, 200, 0), "ADDED")      # Green
        draw_boxes(match_result.get("removed", []), (0, 0, 200), "REMOVED")  # Red
        draw_boxes(match_result.get("moved", []), (255, 0, 0), "MOVED")      # Blue
        draw_boxes(match_result.get("resized", []), (0, 165, 255), "RESIZED") # Orange

        # Construct labeled headers for each composite panel
        def add_header(img, title):
            h, w = img.shape[:2]
            header = np.full((120, w, 3), 40, dtype=np.uint8)
            cv2.putText(header, title, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
            return np.vstack([header, img])

        p1 = add_header(panel_v1, f"V1 (ORIGINAL)")
        p2 = add_header(panel_v2, f"V2 (REVISION)")
        p3 = add_header(canvas_diff, f"ANALYSIS: {drawing_id}")

        report_canvas = np.hstack([p1, p2, p3])

        os.makedirs("visuals", exist_ok=True)
        out = f"visuals/diagram_{drawing_id}.jpg"
        cv2.imwrite(out, report_canvas)
        print(f"   [VISUAL] Saved minimalist diagram to: {out}")
        return out
