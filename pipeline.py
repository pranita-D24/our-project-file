# pipeline.py — v2.1 FIXED
# Fix Bug 4: detect_changes() now receives dim_result so verdict
#            correctly reflects dimension changes even when SSIM is high.

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, time, logging, json
from datetime import datetime
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from config           import IMAGE_DIR, REPORT_DIR
from pdf_processor    import PDFProcessor
from preprocessor     import ImagePreprocessor
from aligner          import ImageAligner
from detector         import ObjectDetector
from matcher          import ObjectMatcher
from change_detector  import ChangeDetector
from report_generator import ReportGenerator
from balloon_filter   import BalloonFilter
from dimension_analyzer import DimensionAnalyzer
from ai_analyzer      import AIAnalyzer
from database         import DatabaseManager

logger = logging.getLogger(__name__)


class ComparisonPipeline:

    def __init__(self):
        self.pdf_proc  = PDFProcessor()
        self.prep      = ImagePreprocessor()
        self.aligner   = ImageAligner()
        self.detector  = ObjectDetector()
        self.matcher   = ObjectMatcher()
        self.changer   = ChangeDetector()
        self.reporter  = ReportGenerator()
        self.balloons  = BalloonFilter()
        self.dims      = DimensionAnalyzer()
        self.ai        = AIAnalyzer()
        self.db        = DatabaseManager()
        logger.info("ComparisonPipeline v2.1 ready")

    # ════════════════════════════════════════
    # REGISTER DRAWING
    # ════════════════════════════════════════
    def register_drawing(self, pdf_path, drawing_name,
                         version_no, drawing_no=None,
                         drawing_id=None, category=None,
                         notes=None, uploaded_by=None):
        try:
            folder_id  = drawing_id or drawing_name.replace(" ", "_")
            pdf_result = self.pdf_proc.process_pdf(pdf_path, folder_id, version_no)
            if not pdf_result["success"]:
                return {"success": False, "errors": pdf_result["errors"]}

            if drawing_id is None:
                drawing_id = self.db.add_drawing(
                    name=drawing_name,
                    drawing_no=drawing_no,
                    category=category)

            version_id = self.db.add_version(
                drawing_id=drawing_id,
                version_no=version_no,
                pdf_path=pdf_result["pdf_path"],
                image_paths=pdf_result["image_paths"],
                page_count=pdf_result["page_count"],
                uploaded_by=uploaded_by,
                notes=notes)

            return {
                "success"    : True,
                "drawing_id" : drawing_id,
                "version_id" : version_id,
                "image_paths": pdf_result["image_paths"],
                "page_count" : pdf_result["page_count"],
                "pdf_path"   : pdf_result["pdf_path"],
            }
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {"success": False, "errors": [str(e)]}

    # ════════════════════════════════════════
    # CORE COMPARISON ENGINE
    # ════════════════════════════════════════
    def _run_comparison_engine(self, img_p1, img_p2,
                                drawing_id="COMPARE",
                                version_1="v1",
                                version_2="v2"):
        import cv2
        import numpy as np
        from config import TARGET_SIZE

        t0 = time.time()

        # ── Load color images for display ──
        orig_color = cv2.imread(img_p1, cv2.IMREAD_COLOR)
        mod_color  = cv2.imread(img_p2, cv2.IMREAD_COLOR)
        if orig_color is not None:
            orig_color = cv2.resize(orig_color, TARGET_SIZE,
                                    interpolation=cv2.INTER_LINEAR)
        if mod_color is not None:
            mod_color = cv2.resize(mod_color, TARGET_SIZE,
                                   interpolation=cv2.INTER_LINEAR)

        # ── Preprocess ──
        orig, mod = self.prep.preprocess_pair(img_p1, img_p2)
        if orig is None or mod is None:
            return {"success": False, "error": "Preprocessing failed"}

        # ── Align ──
        aligned       = self.aligner.align(orig, mod)
        aligned_color = None
        if orig_color is not None and mod_color is not None:
            aligned_color = self.aligner.align(orig_color, mod_color)

        # ── Balloon detection ──
        balloon_regions1 = self.balloons.get_balloon_regions(orig)
        balloon_regions2 = self.balloons.get_balloon_regions(aligned)
        balloon_mask1    = self.balloons.create_balloon_mask(orig.shape, balloon_regions1)
        balloon_mask2    = self.balloons.create_balloon_mask(aligned.shape, balloon_regions2)

        # ── Object detection ──
        objs1 = self.detector.detect_objects(orig,    balloon_mask=balloon_mask1)
        objs2 = self.detector.detect_objects(aligned, balloon_mask=balloon_mask2)

        # ── Match ──
        match_res = self.matcher.match_objects(objs1, objs2)

        # ── Dimension analysis FIRST ──
        # FIX Bug 4: run dims before change detection so we can pass
        # the result in and get an accurate verdict.
        dim_result = self.dims.compare_dimensions(orig, aligned)

        # ── Change detection — now receives dim_result ──
        change_res = self.changer.detect_changes(
            orig, aligned, match_res, dim_result=dim_result)

        if change_res is None:
            return {"success": False, "error": "Change detection failed"}

        # ── AI analysis ──
        ai_result = self.ai.analyze_drawings(orig, aligned)
        if ai_result.get("success"):
            change_res = self.ai.merge_with_cv_results(change_res, ai_result)

        # ── Reports ──
        proc_time = round(time.time() - t0, 2)
        comp_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = self.reporter.generate_all_reports(
            orig_gray       = orig,
            mod_gray        = aligned,
            change_result   = change_res,
            match_result    = match_res,
            dim_result      = dim_result,
            ai_result       = ai_result,
            drawing_id      = drawing_id,
            version_1       = version_1,
            version_2       = version_2,
            comparison_id   = comp_id,
            processing_time = proc_time,
        )

        return {
            "success"          : True,
            "orig"             : orig,
            "aligned"          : aligned,
            "orig_color"       : orig_color,
            "aligned_color"    : aligned_color,
            "change_res"       : change_res,
            "match_res"        : match_res,
            "dim_result"       : dim_result,
            "ai_result"        : ai_result,
            "balloon_regions1" : balloon_regions1,
            "balloon_regions2" : balloon_regions2,
            "report"           : report,
            "processing_time"  : proc_time,
            "objs1_count"      : len(objs1),
            "objs2_count"      : len(objs2),
            "comparison_id"    : comp_id,
        }

    # ════════════════════════════════════════
    # QUICK COMPARE
    # ════════════════════════════════════════
    def quick_compare(self, image_path_1, image_path_2,
                      drawing_name="DRAWING"):
        try:
            result = self._run_comparison_engine(
                image_path_1, image_path_2,
                drawing_id=drawing_name)
            result["success"] = True
            return result
        except Exception as e:
            logger.error(f"Quick compare error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # ════════════════════════════════════════
    # COMPARE PDF VERSIONS
    # ════════════════════════════════════════
    def compare_versions(self, drawing_id,
                         version_1_id, version_2_id,
                         page_no=1):
        try:
            v1 = self.db.get_version(version_1_id)
            v2 = self.db.get_version(version_2_id)
            if not v1 or not v2:
                return {"success": False, "error": "Version not found"}

            paths1 = json.loads(v1.get("image_paths", "[]"))
            paths2 = json.loads(v2.get("image_paths", "[]"))
            if not paths1 or not paths2:
                return {"success": False, "error": "No images found"}

            idx = min(page_no - 1, len(paths1) - 1)
            p1  = paths1[idx]
            p2  = paths2[min(idx, len(paths2) - 1)]

            result = self._run_comparison_engine(
                p1, p2,
                drawing_id=drawing_id,
                version_1=v1["version_no"],
                version_2=v2["version_no"])

            if result["success"]:
                cr = result["change_res"]
                comparison_id = self.db.add_comparison(
                    drawing_id      = drawing_id,
                    version_1_id    = version_1_id,
                    version_2_id    = version_2_id,
                    similarity      = cr["similarity"],
                    added           = cr["added_count"],
                    removed         = cr["removed_count"],
                    modified        = cr["modified_count"],
                    verdict         = cr["verdict"],
                    report_path     = result["report"].get("visual"),
                    json_path       = result["report"].get("json"),
                    processing_time = result["processing_time"])
                result["db_comparison_id"] = comparison_id

            return result

        except Exception as e:
            logger.error(f"Compare versions error: {e}")
            return {"success": False, "error": str(e)}