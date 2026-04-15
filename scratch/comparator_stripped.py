# comparator.py Stripped version

import logging
import time
import os
import fitz
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict

# Stage 4/5/6 Integration
from stage4_geometry import Stage4Engine
from stage2_vector import Stage2Engine
from stage3_balloons import detect_balloons
import stage5_moves
from report_generator import ReportGenerator

logger = logging.getLogger(__name__)

@dataclass
class ChangedRegion:
    x: int; y: int; w: int; h: int
    label: str = ""; change_type: str = "CHANGED"; detail: str = ""

@dataclass
class CompareResult:
    regions: List[ChangedRegion] = field(default_factory=list)
    verdict: str = "IDENTICAL / VERY SIMILAR"
    similarity: float = 100.0
    processing_info: dict = field(default_factory=dict)
    
    # Fidelity fields for reporting
    dim_changes: List[dict] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    drawing_id: str = "Unknown"
    balloons_ignored: int = 0
    processing_time: float = 0.0

def classify_pdf(page: fitz.Page) -> str:
    """Mandatory Type Gate: Distinguishes Vector from Scanned Raster PDFs."""
    text = page.get_text("text").strip()
    drawings = page.get_drawings()
    images = page.get_images()
    
    has_vector = len(drawings) > 0
    has_text = len(text) > 0
    has_images = len(images) > 0
    
    if not has_text and not has_vector and has_images:
        return "RASTER"
    return "VECTOR"

def compare(path1: str, path2: str, sensitivity: float = 0.55, drawing_id: str = "Unknown") -> CompareResult:
    res = CompareResult()
    res.drawing_id = drawing_id
    start_time = time.time()
    
    try:
        doc1, doc2 = fitz.open(path1), fitz.open(path2)
        page1, page2 = doc1[0], doc2[0]
        
        # --- STEP 1: TYPE GATE ---
        t1_type = classify_pdf(page1)
        t2_type = classify_pdf(page2)
        
        if t1_type == "RASTER" or t2_type == "RASTER":
            log_path = r"c:\Trivim Internship\engineering_comparison_system\raster_rejected.log"
            log_msg = f"{datetime.datetime.now()}: Rejected {drawing_id} - RASTER detected (V1: {t1_type}, V2: {t2_type})"
            with open(log_path, "a") as f:
                f.write(log_msg + "\n")
            print(f"\n[TYPE GATE] !!! RASTER REJECTED: {drawing_id}")
            res.verdict = "RASTER_REJECTED"
            return res

        # --- STEP 3: RAW DELTA CHECK (SAFEGUARD) ---
        v1_paths = page1.get_drawings()
        v2_paths = page2.get_drawings()
        path_delta = abs(len(v1_paths) - len(v2_paths)) / max(len(v1_paths), 1)
        
        v1_spans_list = [s for b in page1.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
        v2_spans_list = [s for b in page2.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
        span_delta = abs(len(v1_spans_list) - len(v2_spans_list)) / max(len(v1_spans_list), 1)
        
        prohibit_identical = (path_delta > 0.10) or (span_delta > 0.10)
        if prohibit_identical:
            logger.info(f"[{drawing_id}] Prohibiting IDENTICAL verdict due to raw deltas (Paths: {path_delta:.1%}, Spans: {span_delta:.1%})")

        # --- STEP 4-6: VECTOR ANALYSIS ---
        s2_engine, s4_engine = Stage2Engine(), Stage4Engine()
        t1, t2 = page1.get_text("dict"), page2.get_text("dict")
        b1, b2 = detect_balloons(page1, t1, "V1"), detect_balloons(page2, t2, "V2")
        d1, d2 = s2_engine.extract_page_data(page1, "V1"), s2_engine.extract_page_data(page2, "V2")
        
        # Independent Text Diff (Manual Pass)
        v1_texts = set(s["text"].strip() for s in v1_spans_list if s["text"].strip())
        v2_texts = set(s["text"].strip() for s in v2_spans_list if s["text"].strip())
        removed_texts = v1_texts - v2_texts

        s4_res = s4_engine.compare_pages(
            page1, page2,
            dim_spans_v1=d1["dimensions"], dim_spans_v2=d2["dimensions"],
            balloons_v1=b1["path_indices"], balloons_v2=b2["path_indices"],
            bounds_v1=s2_engine.detect_boundaries(page1), bounds_v2=s2_engine.detect_boundaries(page2)
        )
        
        # --- STEP 7: MOVES & CLUSTERING ---
        s5_raw = stage5_moves.discover_moves(
            added=s4_res["geometry"]["added"], removed=s4_res["geometry"]["removed"],
            page_rect=page2.rect, page_text_dict=page2.get_text("dict")
        )
        
        clustered_added = stage5_moves.cluster_to_components(s5_raw["added"], "ADDED")
        clustered_removed = stage5_moves.cluster_to_components(s5_raw["removed"], "REMOVED")
        
        # --- THE VORONOI ADMIN FILTER (User Request Fix 1) ---
        import re
        # Aggressive regex to catch SCALE and other admin stamps (case-insensitive)
        admin_regex = re.compile(r"DO NOT SCALE|DISTRIBUTION|ALL RIGHTS|RESERVED|CONTROLLED COPY|SCALE", re.IGNORECASE)
        
        def filter_admin_noise(clusters):
            filtered = []
            for c in clusters:
                # Suppress isolated primitive if its region label matches admin regex
                is_isolated = c.get("primitive_count", 1) == 1
                region_label = str(c.get("region", ""))
                
                if is_isolated and admin_regex.search(region_label):
                    print(f"   [ADMIN FILTER] Suppressed noise in region: {c.get('region')}")
                    continue
                filtered.append(c)
            return filtered

        clustered_added = filter_admin_noise(clustered_added)
        clustered_removed = filter_admin_noise(clustered_removed)
        
        # Add explicit Text Deletions to clustered_removed (Independent Text Diff)
        for txt in removed_texts:
            # Only add if it wasn't already matched by geometry (heuristic)
            exists = any(txt in str(r.get("notes", "")) for r in clustered_removed)
            if not exists:
                clustered_removed.append({
                    "type": "text-note", "status": "REMOVED", 
                    "notes": f"Note removed: {txt}", "centroid": [0,0], "primitive_count": 1,
                    "region": "MAIN VIEW" # Default
                })

        # Dimensions logic
        tol = s2_engine.s2_cfg["centroid_match_tolerance_pt"]
        dm, d_rem, d_add = s2_engine.match_entities(d1["dimensions"], d2["dimensions"], tol)
        regions_v2 = stage5_moves.get_drawing_regions(page2.get_text("dict"))
        for e in d_add: e.region = stage5_moves.assign_to_region(e.centroid, regions_v2)
        for e in d_rem: e.region = stage5_moves.assign_to_region(e.centroid, regions_v2)
        d_mod = [{"from": p[0].value, "to": p[1].value, "centroid": p[1].centroid, "region": getattr(p[1], "region", "MAIN VIEW")} for p in dm if p[0].value != p[1].value]

        # --- STEP 8: VERDICT CALCULATION ---
        added_comp = [c for c in clustered_added if "COMPONENT" in c["status"]]
        removed_comp = [c for c in clustered_removed if "COMPONENT" in c["status"]]
        comp_changes = len(added_comp) + len(removed_comp) + len(s4_res["geometry"]["resized"])
        dim_changes_count = len(d_add) + len(d_rem) + len(d_mod)
        
        total_structural = comp_changes + dim_changes_count
        
        if total_structural == 0:
            if prohibit_identical or len(clustered_added) > 5 or len(clustered_removed) > 5:
                res.verdict = "MINOR CHANGES" # Safety floor
            else:
                res.verdict = "IDENTICAL / VERY SIMILAR"
        elif total_structural <= 2:
            res.verdict = "MINOR CHANGES"
        else:
            res.verdict = "MAJOR CHANGES"

        print(f"\n[VERDICT AUDIT] {drawing_id}: Comp:{comp_changes}, Dim:{dim_changes_count} -> {res.verdict}")
        res.similarity = 100.0 if "IDENTICAL" in res.verdict else (85.0 if "MINOR" in res.verdict else 60.0)

        # Excel & Reporting
        xl_data = {
            "drawing_id": drawing_id, "v1_name": os.path.basename(path1), "v2_name": os.path.basename(path2),
            "geometry": {"added": clustered_added, "removed": clustered_removed, "moved": s5_raw["moved"], "resized": s4_res["geometry"]["resized"]},
            "dimensions": {
                "added": [{"value": e.value, "centroid": e.centroid, "region": getattr(e, "region", "MAIN VIEW")} for e in d_add],
                "removed": [{"value": e.value, "centroid": e.centroid, "region": getattr(e, "region", "MAIN VIEW")} for e in d_rem],
                "modified": d_mod
            },
            "balloons_ignored": b2["balloons_ignored"],
            "processing_log": {"PDF Type": "Vector", "Metrics": f"C:{comp_changes} D:{dim_changes_count}", "Path Delta": f"{path_delta:.2f}"}
        }
        
        # Bridge fix (User Request Fix 2: One-line mapping)
        ReportGenerator().generate_all_reports(
            drawing_id=drawing_id, 
            match_result=xl_data["geometry"], 
            dim_result=xl_data["dimensions"], 
            change_result={"similarity": res.similarity, "verdict": res.verdict}, 
            version_1=xl_data["v1_name"], 
            version_2=xl_data["v2_name"],
            balloons_ignored=xl_data["balloons_ignored"]
        )

        res.processing_time = time.time() - start_time
        res.geometry = xl_data["geometry"]
        res.dim_changes = d_mod
        res.balloons_ignored = b1.get("balloons_ignored", 0)
        
        doc1.close(); doc2.close()
        return res
    except Exception as e:
        logger.warning(f"Error Comparing [{drawing_id}]: {e}")
        import traceback; traceback.print_exc()
        return res

def PageCount(doc):
    try: return len(doc)
    except: return 0
