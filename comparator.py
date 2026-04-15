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
from report_generator import ReportGenerator
from reasoning_engine import ReasoningEngine
import yaml
import cv2
import numpy as np
import scipy.spatial
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

@dataclass
class ChangedRegion:
    x: int; y: int; w: int; h: int
    label: str = ""; change_type: str = "CHANGED"; detail: str = ""

@dataclass
class CompareResult:
    regions: List[ChangedRegion] = field(default_factory=list)
    verdict: str = "IDENTICAL / VERY SIMILAR"
    processing_info: dict = field(default_factory=dict)
    
    # Fidelity fields for reporting
    dim_changes: List[dict] = field(default_factory=list)
    geometry: dict = field(default_factory=dict)
    drawing_id: str = "Unknown"
    balloons_ignored: int = 0
    processing_time: float = 0.0
    thumbnail_path: str = ""
    similarity: float = 100.0
    overall_ssim: float = 1.0
    pixel_diff_pct: float = 0.0
    mechanical_story: str = ""
    
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
    if has_vector and not has_text:
        return "GEOMETRY_ONLY"
    return "VECTOR"

def calibrate_geometry_only(drawings):
    """Calibrates match and size tolerances using drawing's own path geometry."""
    centroids = [((d["rect"][0]+d["rect"][2])/2, (d["rect"][1]+d["rect"][3])/2) for d in drawings]
    areas = [abs((d["rect"][2]-d["rect"][0]) * (d["rect"][3]-d["rect"][1])) for d in drawings]
    
    # Match tolerance from spatial density
    if len(centroids) >= 2:
        try:
            tree = scipy.spatial.KDTree(centroids)
            dists, _ = tree.query(centroids, k=min(2, len(centroids)))
            # k=2 because k=1 is the point itself
            match_tolerance = np.median(dists[:, -1]) * 0.3
        except:
            match_tolerance = 15.0
    else:
        # Fallback to page-relative estimate
        if drawings:
            w = abs(drawings[0]["rect"][2] - drawings[0]["rect"][0])
            match_tolerance = w * 0.1
        else:
            match_tolerance = 15.0
    
    # Size tolerance from area distribution
    size_tolerance = np.percentile(areas, 5) if areas else 1.0
    
    # Jitter floor — no text anchors available. Use path bbox precision proxy
    if drawings:
        widths = [abs(d["rect"][2]-d["rect"][0]) for d in drawings]
        jitter_floor = np.percentile(widths, 1) * 0.05
    else:
        jitter_floor = 2.0
        
    match_tolerance = max(match_tolerance, jitter_floor)
    return match_tolerance, size_tolerance

def is_boundary_path(path):
    """
    A path is a boundary (frame/view box) if it matches the 'enclosure signature':
    1. All ops are lines only (no curves like arcs/fillets)
    2. Forms a closed-ish rectangle (4-8 line segments)
    3. Has no fill
    4. Has dimensions consistent with a view box (min 20x20)
    5. Aspect ratio is balanced (not a line or leader)
    """
    items = path.get("items", [])
    ops = [it[0] for it in items]
    
    # Must be all lines (curves like 'c' or 'qu' imply structural geometry)
    if not ops or not all(op == 'l' for op in ops):
        return False
    
    # Must be rectangular/chamfered box (4-8 segments)
    if not (4 <= len(ops) <= 8):
        return False
    
    # Boundaries are typically strokes, not fills
    if path.get("fill") is not None:
        return False
    
    # Size check — must be large enough to be a box (not a marker/hatch)
    r = path["rect"]
    if r.width < 20 or r.height < 20:
        return False
    
    # Aspect ratio check — must be boxy (not a thin horizontal/vertical line)
    ratio = r.width / r.height if r.height > 0 else 999
    if not (0.2 <= ratio <= 5.0):
        return False
    
    return True

def is_administrative_zone(bbox, page_width: float, page_height: float) -> bool:
    """
    Returns True if this bounding box looks like a title block / admin frame.
    bbox = (x0, y0, x1, y1) or fitz.Rect in PDF points
    """
    x0, y0, x1, y1 = bbox
    zone_width  = x1 - x0
    zone_height = y1 - y0

    width_ratio  = zone_width  / page_width
    bottom_start = y0 / page_height

    # Title blocks: wide (>75% of page), in bottom 30% of page
    if width_ratio > 0.75 and bottom_start > 0.70:
        return True

    # Also catch very thin tall strips on left/right (border lines)
    if zone_width < 20 or zone_height < 20:
        return True

    return False

def filter_non_structural(drawings, outer_frame, page_rect):
    """
    Cluster-Based Filter with Administrative Gating:
    1. Only preserves paths inside the outer cluster frame.
    2. Explicitly purges paths matching the is_administrative_zone signature.
    """
    if not drawings:
        return drawings
    
    f = outer_frame
    structural = []
    for d in drawings:
        r = d["rect"]
        # Skip if administrative (Title Block lines)
        if is_administrative_zone(r, page_rect.width, page_rect.height):
            continue
            
        cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
        if f[0]-2 <= cx <= f[2]+2 and f[1]-2 <= cy <= f[3]+2:
            # Also exclude the frame itself (very large rects)
            if (r.width * r.height) < (f.width * f.height * 0.95):
                structural.append(d)
    
    if not structural:
        return drawings
        
    return structural

def generate_six_panel_report(v1_path, v2_path, result, output_dir, drawing_id):
    """
    Professional Forensic Layout:
    Row 1: Original | Modified | Difference View (Precise Markers)
    Row 2: SSIM Heatmap | Pixel Diff Map | Stats Summary
    + Bottom Metadata Strip
    """
    PDF_DPI = 72
    RENDER_DPI = 300
    SCALE = RENDER_DPI / PDF_DPI 

    def pdf_pt_to_px(pt_coord: tuple[float, float]) -> tuple[int, int]:
        return int(pt_coord[0] * SCALE), int(pt_coord[1] * SCALE)

    # 1. Render Source Pages (300 DPI for high-fidelity)
    doc1, doc2 = fitz.open(v1_path), fitz.open(v2_path)
    mat = fitz.Matrix(SCALE, SCALE) # Exactly synced with SCALE
    pix1, pix2 = doc1[0].get_pixmap(matrix=mat), doc2[0].get_pixmap(matrix=mat)
    
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.height, pix1.width, pix1.n)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.height, pix2.width, pix2.n)
    
    # Ensure 3-channel BGR
    if pix1.n == 4: img1 = cv2.cvtColor(img1, cv2.COLOR_RGBA2BGR)
    else: img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    if pix2.n == 4: img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2BGR)
    else: img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    
    # Standardize size for the grid (Panel Width = 1000px)
    target_w = 1000
    target_h = int(img1.shape[0] * (target_w / img1.shape[1]))
    
    p1 = cv2.resize(img1, (target_w, target_h))
    p2 = cv2.resize(img2, (target_w, target_h))
    
    # Panel 3: Precise Markers (Drawn on V2 base)
    p3 = p2.copy()
    
    # Geometry — Small Precise Squares at Centroids (Green=Add, Red=Rem)
    marker_size = 12
    for change in result.geometry.get("added", []):
        if "centroid" in change:
            cx, cy = pdf_pt_to_px(change["centroid"])
            cv2.rectangle(p3, (cx-marker_size, cy-marker_size), (cx+marker_size, cy+marker_size), (0,200,0), -1)
            cv2.rectangle(p3, (cx-marker_size, cy-marker_size), (cx+marker_size, cy+marker_size), (0,100,0), 2)
            
    for change in result.geometry.get("removed", []):
        if "centroid" in change:
            cx, cy = pdf_pt_to_px(change["centroid"])
            cv2.rectangle(p3, (cx-marker_size, cy-marker_size), (cx+marker_size, cy+marker_size), (0,0,200), -1)
            cv2.rectangle(p3, (cx-marker_size, cy-marker_size), (cx+marker_size, cy+marker_size), (0,0,100), 2)
            
    # Dimensions — Precise Circles (Yellow=Mod, Green=Add, Red=Rem)
    for d_type, color in [("added", (0,200,0)), ("removed", (0,0,200)), ("modified", (0,255,255))]:
        target_list = result.dim_changes if d_type == "modified" else result.processing_info.get("dimensions", {}).get(d_type, [])
        for d in target_list:
            if "centroid" in d:
                cx, cy = pdf_pt_to_px(d["centroid"])
                cv2.circle(p3, (cx, cy), 18, color, 3)
                v_str = f"{d.get('from')}->{d.get('to')}" if d_type == "modified" else str(d.get("value", ""))
                cv2.putText(p3, v_str, (cx+22, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Panel 4: SSIM Heatmap (Skimage)
    g1 = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)
    score, ssim_map = structural_similarity(g1, g2, full=True)
    ssim_view = (ssim_map * 255).astype(np.uint8)
    p4 = cv2.applyColorMap(ssim_view, cv2.COLORMAP_JET)
    
    # Panel 5: Pixel Diff Map (OpenCV)
    pixel_diff = cv2.absdiff(g1, g2)
    p5 = cv2.applyColorMap(pixel_diff, cv2.COLORMAP_HOT)
    
    # Panel 6: Stats & Log
    p6 = np.full((target_h, target_w, 3), (40, 40, 40), dtype=np.uint8)
    changed_pixels = np.count_nonzero(pixel_diff > 30)
    total_pixels = g1.size
    changed_pct = (changed_pixels / total_pixels) * 100
    
    # Stats Bar
    bar_w, bar_h = 800, 60
    bx, by = 100, 200
    cv2.rectangle(p6, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
    changed_w = int(bar_w * (changed_pct/100))
    cv2.rectangle(p6, (bx, by), (bx + (bar_w - changed_w), by + bar_h), (0, 150, 0), -1) # Green = Unchanged
    cv2.rectangle(p6, (bx + (bar_w - changed_w), by), (bx + bar_w, by + bar_h), (0, 0, 150), -1) # Red = Changed
    
    cv2.putText(p6, f"Pixel Change Sensitivity: {changed_pct:.2f}%", (bx, by - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Panel Labels
    for i, (p, txt) in enumerate([(p1, "ORIGINAL (V1)"), (p2, "MODIFIED (V2)"), (p3, "VECTOR AUDIT"), (p4, "SSIM HEATMAP"), (p5, "PIXEL DIFF"), (p6, "STATS SURVEY")]):
        cv2.putText(p, txt, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 6) # Border
        cv2.putText(p, txt, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # 4. Construct Grid
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid = np.vstack([row1, row2])
    
    # 5. Bottom Metadata Strip
    footer_h = 150
    footer = np.full((footer_h, grid.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    f_txt = f"ID: {drawing_id} | Verdict: {result.verdict} | Add: {len(result.geometry.get('added', []))} | Rem: {len(result.geometry.get('removed', []))} | Time: {result.processing_time:.2f}s"
    cv2.putText(footer, f_txt, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
    
    final_report = np.vstack([grid, footer])
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{drawing_id}_diff.png")
    cv2.imwrite(out_path, final_report)
    print(f"   [SIX-PANEL REPORT] Generated: {out_path}")
    doc1.close(); doc2.close()
    return out_path

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

        # --- STEP 2: DYNAMIC BOUNDARIES ---
        s2_engine = Stage2Engine()
        b1_bounds = s2_engine.detect_boundaries(page1)
        b2_bounds = s2_engine.detect_boundaries(page2)
        
        # --- STEP 3: RAW DELTA CHECK (SAFEGUARD) ---
        v1_paths = filter_non_structural(page1.get_drawings(), b1_bounds["outer_frame"], page1.rect)
        v2_paths = filter_non_structural(page2.get_drawings(), b2_bounds["outer_frame"], page2.rect)
        path_delta = abs(len(v1_paths) - len(v2_paths)) / max(len(v1_paths), 1)
        
        v1_spans_list = [s for b in page1.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
        v2_spans_list = [s for b in page2.get_text("dict")["blocks"] if "lines" in b for l in b["lines"] for s in l["spans"]]
        span_delta = abs(len(v1_spans_list) - len(v2_spans_list)) / max(len(v1_spans_list), 1)
        
        prohibit_identical = (path_delta > 0.10) or (span_delta > 0.10)
        if prohibit_identical:
            logger.info(f"[{drawing_id}] Prohibiting IDENTICAL verdict due to raw deltas (Paths: {path_delta:.1%}, Spans: {span_delta:.1%})")

        # --- STEP 4-6: VECTOR ANALYSIS ---
        with open("pipeline_config.yaml", "r") as f:
            config = yaml.safe_load(f)

        s2_engine, s4_engine = Stage2Engine(), Stage4Engine()
        
        # Override Stage4 tolerance if GEOMETRY_ONLY
        if t1_type == "GEOMETRY_ONLY" or t2_type == "GEOMETRY_ONLY":
            logger.info(f"[{drawing_id}] Activating GEOMETRY_ONLY mode (No Spans Detected)")
            match_tol, size_tol = calibrate_geometry_only(v1_paths)
            s4_engine.match_tolerance = match_tol
            # Note: size_tol can also be used if Stage4 is updated to support it
        
        t1, t2 = page1.get_text("dict"), page2.get_text("dict")
        b1, b2 = detect_balloons(page1, t1, "V1", drawings=v1_paths), detect_balloons(page2, t2, "V2", drawings=v2_paths)
        d1, d2 = s2_engine.extract_page_data(page1, "V1"), s2_engine.extract_page_data(page2, "V2")
        
        # Skip Dimension Processing if Geometry-Only (No readable dimensions)
        if t1_type == "GEOMETRY_ONLY":
            d1["dimensions"] = []
            d2["dimensions"] = []

        # Independent Text Diff (Manual Pass)
        v1_texts = set(s["text"].strip() for s in v1_spans_list if s["text"].strip())
        v2_texts = set(s["text"].strip() for s in v2_spans_list if s["text"].strip())
        removed_texts = v1_texts - v2_texts

        s4_res = s4_engine.compare_pages(
            page1, page2,
            dim_spans_v1=d1["dimensions"], dim_spans_v2=d2["dimensions"],
            balloons_v1=b1["path_indices"], balloons_v2=b2["path_indices"],
            bounds_v1=s2_engine.detect_boundaries(page1), bounds_v2=s2_engine.detect_boundaries(page2),
            drawings_v1=v1_paths, drawings_v2=v2_paths
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

        # --- STEP 7: DIMENSION COMPARISON (Set-Diff by Centroid) ---
        def match_dim_set(dims1, dims2):
            matches, rem, add = [], [], []
            matched_v2 = set()
            
            for d1 in dims1:
                found = False
                # Epsilon from d1's median height (dynamic)
                eps = d1.median_height * 1.5 if hasattr(d1, 'median_height') else 15.0
                
                for i, d2 in enumerate(dims2):
                    if i in matched_v2: continue
                    # Value similarity (ignore whitespace/special characters)
                    v1_clean = re.sub(r'[\s\xad]', '', d1.value)
                    v2_clean = re.sub(r'[\s\xad]', '', d2.value)
                    
                    if v1_clean == v2_clean:
                        dist = ((d1.centroid[0]-d2.centroid[0])**2 + (d1.centroid[1]-d2.centroid[1])**2)**0.5
                        if dist < eps:
                            matches.append((d1, d2))
                            matched_v2.add(i)
                            found = True
                            break
                if not found:
                    rem.append(d1)
            
            add = [dims2[i] for i in range(len(dims2)) if i not in matched_v2]
            return matches, rem, add

        dm, d_rem, d_add = match_dim_set(d1["dimensions"], d2["dimensions"])
        regions_v2 = stage5_moves.get_drawing_regions(page2.get_text("dict"))
        for e in d_add: e.region = stage5_moves.assign_to_region(e.centroid, regions_v2)
        for e in d_rem: e.region = stage5_moves.assign_to_region(e.centroid, regions_v2)
        d_mod = [{"from": p[0].value, "to": p[1].value, "centroid": p[1].centroid, "region": getattr(p[1], "region", "MAIN VIEW")} for p in dm if p[0].value != p[1].value]

        # --- STEP 8: VERDICT CALCULATION (TIERED CALIBRATION) ---
        added_comp = [c for c in clustered_added if "COMPONENT" in c["status"]]
        removed_comp = [c for c in clustered_removed if "COMPONENT" in c["status"]]
        comp_changes = len(added_comp) + len(removed_comp) + len(s4_res["geometry"]["resized"])
        dim_changes_count = len(d_add) + len(d_rem) + len(d_mod)
        total_structural = comp_changes + dim_changes_count

        # FIX 1: Verdict must use counts, not pixel sensitivity alone
        if total_structural == 0:
            res.verdict = "NO CHANGE"
        elif total_structural <= 10:
            res.verdict = "MINOR CHANGES"
        elif total_structural <= 30:
            res.verdict = "MODERATE CHANGES"
        else:
            res.verdict = "MAJOR CHANGES"



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
        
        # Populate result BEFORE thumbnail/reports
        res.geometry = xl_data["geometry"]
        res.dim_changes = d_mod
        res.balloons_ignored = b1.get("balloons_ignored", 0)
        res.processing_info["dimensions"] = xl_data["dimensions"]

        # Visual Diff Thumbnails
        if False and config.get("thumbnails", {}).get("enabled"):
            try:
                # Track final processing time for header
                res.processing_time = time.time() - start_time
                
                thumb_path = generate_six_panel_report(
                    path1, path2, res, 
                    config["thumbnails"]["output_dir"], 
                    drawing_id
                )
                res.thumbnail_path = thumb_path
                xl_data["thumbnail_path"] = thumb_path
            except Exception as ethumb:
                logger.warning(f"Failed to generate thumbnail for {drawing_id}: {ethumb}")

        # --- STEP 9: REASONING ENGINE (Human Intelligence) ---
        all_changes = clustered_added + clustered_removed + s4_res["geometry"]["resized"]
        engine = ReasoningEngine()
        audit_res = engine.run_full_audit(all_changes)
        res.mechanical_story = audit_res.get("mechanical_story", "")

        res.processing_time = time.time() - start_time
        
        doc1.close(); doc2.close()
        return res
    except Exception as e:
        logger.warning(f"Error Comparing [{drawing_id}]: {e}")
        import traceback; traceback.print_exc()
        return res

def PageCount(doc):
    try: return len(doc)
    except: return 0

def simple_report(result):
    g = result.geometry if hasattr(result, 'geometry') and result.geometry else {}
    print("="*40)
    print(f"ADDED:   {len(g.get('added', []))}")
    print(f"REMOVED: {len(g.get('removed', []))}")
    print(f"MOVED:   {len(g.get('moved', []))}")
    print(f"RESIZED: {len(g.get('resized', []))}")
    print("="*40)
