# pdf_reader.py — Per-Drawing Adaptive Profile Reader
# Phase 1 of the Adaptive Intelligence Architecture.
# Extracts text, detects layout, and caches a DrawingProfile
# for any engineering drawing PDF — regardless of standard or format.

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = pathlib.Path("./profile_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────
# DRAWING PROFILE DATACLASS
# ────────────────────────────────────────────────────────────
@dataclass
class DrawingProfile:
    # Identity
    drawing_number:   str = ""
    revision:         str = ""
    drawing_standard: str = "UNKNOWN"  # ISO / ANSI / DIN / JIS / AS / BS
    units:            str = "mm"
    scale:            str = "1:1"
    scale_ratio:      float = 1.0      # numeric e.g. 0.5 for 1:2

    # Layout geometry (pixel coords at 300 DPI)
    title_block_bbox:  Tuple = (0, 0, 0, 0)   # (x1,y1,x2,y2)
    border_bbox:       Tuple = (0, 0, 0, 0)
    content_bbox:      Tuple = (0, 0, 0, 0)
    gear_data_bbox:    Tuple = (0, 0, 0, 0)
    revision_table_bbox: Tuple = (0, 0, 0, 0)

    # Drawing characteristics
    has_gear_data_table: bool = False
    has_section_views:   bool = False
    has_detail_views:    bool = False
    has_revision_table:  bool = False
    estimated_complexity: str = "moderate"

    # Adaptive thresholds — derived from this drawing
    min_component_area:   int   = 500
    max_component_area:   int   = 1000000
    dim_line_min_length:  int   = 30
    balloon_radius_min:   int   = 12
    balloon_radius_max:   int   = 65
    move_threshold_px:    float = 15.0
    ssim_threshold:       float = 0.999  # never changes

    # Image shape at 300 DPI
    image_width:  int = 0
    image_height: int = 0

    # Vision LLM detected zones
    _isometric_bboxes: list = field(default_factory=list)
    _notes_zones: list = field(default_factory=list)


# ────────────────────────────────────────────────────────────
# MD5 HASH  (for caching)
# ────────────────────────────────────────────────────────────
def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ────────────────────────────────────────────────────────────
# PDF → IMAGE  (300 DPI)
# ────────────────────────────────────────────────────────────
def pdf_to_image(path: str, dpi: int = 300) -> Optional[np.ndarray]:
    """
    Render first page of PDF to a numpy BGR array at *dpi*.
    Falls back to cv2.imread for image files.
    """
    ext = pathlib.Path(path).suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        img = cv2.imread(path)
        return img

    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(path)
        page = doc.load_page(0)
        mat  = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix  = page.get_pixmap(matrix=mat, alpha=False)
        arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        if pix.n == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return arr
    except Exception as e:
        logger.warning(f"pdf_to_image failed for {path}: {e}")
        return None


# ────────────────────────────────────────────────────────────
# TEXT EXTRACTION  (PyMuPDF + pdfplumber)
# ────────────────────────────────────────────────────────────
def _extract_text_pdfplumber(path: str) -> str:
    """
    Thread B: extract embedded text from page 0 using pdfplumber.
    Returns empty string on any error — never raises.
    """
    ext = pathlib.Path(path).suffix.lower()
    if ext not in (".pdf",):
        return ""
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            if not pdf.pages:
                return ""
            return pdf.pages[0].extract_text() or ""
    except Exception as e:
        logger.debug(f"pdfplumber extraction failed: {e}")
        return ""


def extract_text(path: str) -> str:
    """Extract all text from a PDF using embedded layer (PyMuPDF), OCR fallback."""
    ext = pathlib.Path(path).suffix.lower()

    # Image files — go straight to OCR
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        return _ocr_text(path)

    text = ""
    try:
        import fitz
        doc  = fitz.open(path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        logger.warning(f"fitz text extraction failed: {e}")

    if not text.strip():
        logger.info("No embedded text — falling back to OCR")
        text = _ocr_text(path)

    return text


def _ocr_text(path: str) -> str:
    try:
        import pytesseract
        img  = pdf_to_image(path)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray, config="--psm 6 --oem 3")
    except Exception as e:
        logger.warning(f"OCR fallback failed: {e}")
        return ""


# ────────────────────────────────────────────────────────────
# STANDARD DETECTION
# ────────────────────────────────────────────────────────────
_STANDARD_KEYWORDS = {
    "ISO":  ["ISO 128", "ISO128", "ISO 2768", "ISO", "BS EN ISO"],
    "DIN":  ["DIN 128", "DIN", "DIN ISO"],
    "ANSI": ["ANSI", "ASME", "ASME Y14", "ANSI Y14"],
    "JIS":  ["JIS", "JIS B"],
    "BS":   ["BS", "BS 308", "BS8888"],
    "AS":   ["AS 1100", "AS1100", "AS "],
}

def detect_standard(text: str) -> str:
    upper = text.upper()
    for std, keywords in _STANDARD_KEYWORDS.items():
        for kw in keywords:
            if kw.upper() in upper:
                return std
    return "UNKNOWN"


# ────────────────────────────────────────────────────────────
# SCALE & UNIT EXTRACTION
# ────────────────────────────────────────────────────────────
_SCALE_PATTERNS = [
    r'(?:SCALE\s*[:\-]?\s*)(\d+)\s*[:/]\s*(\d+)',   # SCALE 1:2 or 1/2
    r'(\d+)\s*:\s*(\d+)',                              # bare 1:2
    r'(\d+)\s*/\s*(\d+)',                              # 1/2
]

_UNIT_PATTERNS = {
    "mm"  : [r'\bmm\b', r'millim'],
    "inch": [r'\binch\b', r'\bin\b', r'["\u2033]', r'\binches\b'],
    "dual": [r'\bmm\b.*["\u2033]', r'["\u2033].*\bmm\b'],
}

def detect_scale_and_units(text: str) -> Tuple[str, float, str]:
    """Returns (scale_str, scale_ratio, units)."""
    upper = text.upper()

    # Units
    units = "mm"
    for u, pats in _UNIT_PATTERNS.items():
        for p in pats:
            if re.search(p, text, re.IGNORECASE):
                units = u
                break

    # Scale
    for pat in _SCALE_PATTERNS:
        m = re.search(pat, upper)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b > 0:
                ratio = a / b
                return f"{a}:{b}", ratio, units
    return "1:1", 1.0, units


# ────────────────────────────────────────────────────────────
# DRAWING CHARACTERISTICS
# ────────────────────────────────────────────────────────────
_GEAR_KEYWORDS   = ["GEAR DATA", "GEAR TABLE", "TOOTH DATA", "MODULE", "HELIX ANGLE"]
_SECTION_KWORDS  = ["SECTION", "SEC ", "S-S", "A-A", "B-B", "C-C"]
_DETAIL_KWORDS   = ["DETAIL", "VIEW ", "DET "]
_REVISION_KWORDS = ["REV", "REVISION", "ECO", "ECN", "CHANGE"]

def detect_characteristics(text: str) -> dict:
    upper = text.upper()
    return {
        "has_gear_data_table": any(k in upper for k in _GEAR_KEYWORDS),
        "has_section_views":   any(k in upper for k in _SECTION_KWORDS),
        "has_detail_views":    any(k in upper for k in _DETAIL_KWORDS),
        "has_revision_table":  any(k in upper for k in _REVISION_KWORDS),
    }


# ────────────────────────────────────────────────────────────
# ADAPTIVE THRESHOLDS (from scale)
# ────────────────────────────────────────────────────────────
_BASE_MIN_AREA  = 500
_BASE_MAX_AREA  = 1000000
_BASE_DIM_LEN   = 30
_BASE_MOVE_PX   = 15.0

def compute_adaptive_thresholds(scale_ratio: float, image_w: int, image_h: int) -> dict:
    """
    Derive per-drawing thresholds from scale ratio.
    scale_ratio = numerator / denominator  e.g.  1:2 → 0.5, 2:1 → 2.0
    """
    if scale_ratio <= 0:
        scale_ratio = 1.0
    # Clamp to avoid extreme values
    scale_ratio = max(0.1, min(scale_ratio, 10.0))

    min_area = _BASE_MIN_AREA
    max_area = _BASE_MAX_AREA
    dim_len  = int(_BASE_DIM_LEN  * scale_ratio)
    move_px  = _BASE_MOVE_PX / scale_ratio

    # Balloon radius from image size (proxy for text height)
    text_h_px = max(8, int(image_h * 0.007))    # ~0.7% of image height
    bal_min   = max(6,  text_h_px)
    bal_max   = max(25, text_h_px * 5)

    # Clamp to sane ranges
    min_area = max(200, min(min_area, 5000))
    max_area = max(50000, min(max_area, 5_000_000))
    dim_len  = max(15, min(dim_len, 200))
    move_px  = max(5.0, min(move_px, 80.0))
    bal_min  = max(6,  min(bal_min, 30))
    bal_max  = max(20, min(bal_max, 120))

    return {
        "min_component_area":  min_area,
        "max_component_area":  max_area,
        "dim_line_min_length": dim_len,
        "balloon_radius_min":  bal_min,
        "balloon_radius_max":  bal_max,
        "move_threshold_px":   round(move_px, 1),
    }


# ────────────────────────────────────────────────────────────
# DRAWING NUMBER / REVISION EXTRACTION
# ────────────────────────────────────────────────────────────
_DRW_NUM_PAT = re.compile(
    r'(?:DWG|DRG|DRAWING|PART\s*NO|PART\s*NUMBER)[:\s#]*([A-Z0-9\-/\.]+)',
    re.IGNORECASE)
_REV_PAT = re.compile(
    r'\bREV(?:ISION)?\.?\s*[:\-]?\s*([A-Z0-9\.]+)', re.IGNORECASE)

def extract_identity(text: str) -> Tuple[str, str]:
    """Returns (drawing_number, revision)."""
    drw = ""
    m = _DRW_NUM_PAT.search(text)
    if m:
        drw = m.group(1).strip()

    rev = ""
    m = _REV_PAT.search(text)
    if m:
        rev = m.group(1).strip()

    return drw, rev


# ────────────────────────────────────────────────────────────
# COMPLEXITY ESTIMATE
# ────────────────────────────────────────────────────────────
def estimate_complexity(profile: DrawingProfile, image: Optional[np.ndarray]) -> str:
    score = 0
    if profile.has_gear_data_table: score += 2
    if profile.has_section_views:   score += 1
    if profile.has_detail_views:    score += 1
    if profile.has_revision_table:  score += 1
    if image is not None:
        # Edge density proxy
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges  = cv2.Canny(gray, 30, 90)
        density = float(np.mean(edges > 0))
        if density > 0.20: score += 2
        elif density > 0.10: score += 1
    if score >= 4: return "complex"
    if score >= 2: return "moderate"
    return "simple"


# ────────────────────────────────────────────────────────────
# OPENCV PREPROCESSING  (Step 4 of ingestion pipeline)
# ────────────────────────────────────────────────────────────
def _preprocess_for_layout(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the 4-step CV preprocessing pipeline before layout detection:
      1. Deskew     — Hough-line rotation correction
      2. Denoise    — fastNlMeans (preserves thin CAD lines)
      3. CLAHE      — contrast enhancement (L-channel in LAB)
      4. Otsu       — binarize for layout detection

    Returns
    -------
    preprocessed_bgr : np.ndarray
        Deskewed + denoised + CLAHE image. Pass to detect_layout().
    binary_gray : np.ndarray
        Otsu binary (THRESH_BINARY_INV). Store on profile._preprocessed_binary
        for downstream component extraction — do NOT discard.
    """
    try:
        import preprocessor as _pp_mod

        # 1. Deskew (Hough correction — up to ~5°)
        # Instantiate only to call deskew() — avoids banner log with emoji chars
        _pp = _pp_mod.ImagePreprocessor.__new__(_pp_mod.ImagePreprocessor)
        img = _pp.deskew(image_bgr)

        # 2. Denoise — fastNlMeansDenoising on BGR keeps thin lines
        if len(img.shape) == 3:
            img = cv2.fastNlMeansDenoisingColored(
                img, None, h=5, hColor=5,
                templateWindowSize=7, searchWindowSize=21)
        else:
            img = cv2.fastNlMeansDenoising(
                img, None, h=5,
                templateWindowSize=7, searchWindowSize=21)

        # 3. CLAHE on L-channel (LAB) for colour images, gray channel otherwise
        if len(img.shape) == 3:
            lab   = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            l     = clahe.apply(l)
            img   = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray  = clahe.apply(img)
            img   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 4. Otsu binarize — for layout detection and component extraction
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return img, binary

    except Exception as e:
        logger.warning(f"_preprocess_for_layout failed, returning raw image: {e}")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
            if len(image_bgr.shape) == 3 else image_bgr
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return image_bgr, binary


# ────────────────────────────────────────────────────────────
# MAIN READ & PROFILE
# ────────────────────────────────────────────────────────────
def read_and_profile(pdf_path: str) -> DrawingProfile:
    """
    Full profile extraction for a single drawing file.
    Step 3: PyMuPDF raster + pdfplumber text run in parallel (ThreadPoolExecutor).
    Step 4: OpenCV preprocessing before layout detection.
    Steps 5-6: layout detection on preprocessed image, then profile assembly.
    """
    from layout_detector import detect_layout

    profile = DrawingProfile()

    ext = pathlib.Path(pdf_path).suffix.lower()
    is_pdf = ext == ".pdf"

    if is_pdf:
        # ── Step 3: Parallel extraction ──────────────────────────
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_img  = ex.submit(pdf_to_image, pdf_path, 300)       # Thread A: PyMuPDF
            fut_text = ex.submit(_extract_text_pdfplumber, pdf_path) # Thread B: pdfplumber
        image        = fut_img.result()
        plumber_text = fut_text.result()

        # Text merge rule: pick exactly ONE source, never concatenate
        if len(plumber_text.strip()) > 50:
            text = plumber_text
            logger.info("Text source: pdfplumber")
        else:
            text = extract_text(pdf_path)  # PyMuPDF / OCR fallback
            logger.info("Text source: PyMuPDF/OCR fallback")
    else:
        # Image file — single sequential load + no pdfplumber
        image = pdf_to_image(pdf_path)
        text  = extract_text(pdf_path)

    # Identity
    profile.drawing_number, profile.revision = extract_identity(text)

    # Standard
    profile.drawing_standard = detect_standard(text)

    # Scale & units
    profile.scale, profile.scale_ratio, profile.units = detect_scale_and_units(text)

    # Characteristics
    chars = detect_characteristics(text)
    profile.has_gear_data_table = chars["has_gear_data_table"]
    profile.has_section_views   = chars["has_section_views"]
    profile.has_detail_views    = chars["has_detail_views"]
    profile.has_revision_table  = chars["has_revision_table"]

    if image is not None:
        profile.image_height, profile.image_width = image.shape[:2]

        # ── Step 4: OpenCV preprocessing ─────────────────────────
        preprocessed_bgr, binary_gray = _preprocess_for_layout(image)

        # ── Step 5: Layout detection on preprocessed image ───────
        # detect_layout() accepts BGR and re-binarises internally.
        # Pass preprocessed_bgr (not the raw image or the binary).
        layout = detect_layout(preprocessed_bgr)

        # [STAGE 2 HANDOFF] Store binary for downstream component extraction.
        # Future optimization: component extraction and YOLO should consume 
        # profile._preprocessed_binary instead of re-binarizing independently.
        profile._preprocessed_binary = binary_gray

        # Cast to int to prevent string-type pollution in downstream logic
        profile.title_block_bbox    = tuple(map(int, layout["title_block_bbox"]))
        profile.border_bbox         = tuple(map(int, layout["border_bbox"]))
        profile.content_bbox        = tuple(map(int, layout["content_bbox"]))
        profile.gear_data_bbox      = tuple(map(int, layout.get("gear_data_bbox", (0, 0, 0, 0))))
        profile.revision_table_bbox = tuple(map(int, layout.get("revision_table_bbox", (0, 0, 0, 0))))

        # Bug 2: Title block fallback for undetected regions
        if profile.title_block_bbox == (0, 0, 0, 0):
            h_img, w_img = image.shape[:2]
            profile.title_block_bbox = (int(w_img * 0.4), int(h_img * 0.78), int(w_img), int(h_img))
            logger.info(f"Handled title block fallback: {profile.title_block_bbox}")

        # Override gear/revision detect if layout found them
        if profile.gear_data_bbox != (0, 0, 0, 0):
            profile.has_gear_data_table = True
        if profile.revision_table_bbox != (0, 0, 0, 0):
            profile.has_revision_table = True

        # Adaptive thresholds
        thresholds = compute_adaptive_thresholds(
            profile.scale_ratio, profile.image_width, profile.image_height)
        for k, v in thresholds.items():
            setattr(profile, k, v)

        # Balloon radius from contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        profile.balloon_radius_min, profile.balloon_radius_max = _estimate_balloon_range(gray)

        # Use preprocessed image for complexity estimation
        profile.estimated_complexity = estimate_complexity(profile, preprocessed_bgr)
    else:
        # No image available — fallback thresholds
        thresholds = compute_adaptive_thresholds(profile.scale_ratio, 2480, 3508)
        for k, v in thresholds.items():
            setattr(profile, k, v)

    logger.info(
        f"Profile: std={profile.drawing_standard} scale={profile.scale} "
        f"units={profile.units} complexity={profile.estimated_complexity} "
        f"title_block={profile.title_block_bbox}"
    )
    return profile

def _estimate_balloon_range(gray):
    # adaptiveThreshold \u2192 find circular contours \u2192 measure radii
    thresh = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    radii = []
    for c in contours:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        if peri == 0: continue
        circularity = 4 * np.pi * area / (peri**2)
        if circularity > 0.75:
            (_, _), r = cv2.minEnclosingCircle(c)
            if 8 < r < 60:
                radii.append(r)
    if len(radii) < 3:
        H = gray.shape[0]
        return int(H * 0.007), int(H * 0.025)   # safe fallback
    return int(np.percentile(radii, 10) * 0.8), int(np.percentile(radii, 90) * 1.2)


# ────────────────────────────────────────────────────────────
# CACHING
# ────────────────────────────────────────────────────────────
def get_or_create_profile(pdf_path: str) -> DrawingProfile:
    """
    Returns a cached DrawingProfile if available, else creates and caches it.
    Each PDF is profiled exactly once, even across 270,000 pairs.
    """
    try:
        fhash     = _file_hash(pdf_path)
        cache_file = CACHE_DIR / f"{fhash}.json"

        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            # Convert list tuples back
            for k in ("title_block_bbox", "border_bbox", "content_bbox",
                      "gear_data_bbox", "revision_table_bbox"):
                if k in data and isinstance(data[k], list):
                    data[k] = tuple(data[k])
            logger.info(f"Profile cache HIT: {pathlib.Path(pdf_path).name}")
            return DrawingProfile(**data)

        logger.info(f"Profile cache MISS: {pathlib.Path(pdf_path).name} \u2014 profiling...")
        profile = read_and_profile(pdf_path)
        
        try:
            from agent_verifier import calibrate_profile_with_vision
            img_bgr = pdf_to_image(pdf_path)
            if img_bgr is not None:
                calibrate_profile_with_vision(img_bgr, profile)
        except Exception as e:
            logger.warning(f"Vision calibration skipped: {e}")

        cache_file.write_text(
            json.dumps(asdict(profile), default=str, indent=2),
            encoding="utf-8"
        )
        return profile

    except Exception as e:
        logger.error(f"get_or_create_profile failed for {pdf_path}: {e}")
        return DrawingProfile()


# ────────────────────────────────────────────────────────────
# PROFILE MERGING (for V1 + V2 pair)
# ────────────────────────────────────────────────────────────
def merge_profiles(p1: DrawingProfile, p2: DrawingProfile) -> DrawingProfile:
    """
    Creates a merged profile used during the comparison.
    Conservative: takes the more restrictive thresholds to avoid missing changes.
    """
    merged = DrawingProfile()

    # Standard: prefer explicit over UNKNOWN
    merged.drawing_standard = p1.drawing_standard if p1.drawing_standard != "UNKNOWN" else p2.drawing_standard
    merged.units             = p1.units
    merged.scale             = p1.scale
    merged.scale_ratio       = p1.scale_ratio

    # Layout: use V1 as reference (V2 is aligned to V1)
    merged.title_block_bbox    = p1.title_block_bbox
    merged.border_bbox         = p1.border_bbox
    merged.content_bbox        = p1.content_bbox
    merged.gear_data_bbox      = p1.gear_data_bbox
    merged.revision_table_bbox = p1.revision_table_bbox

    # Characteristics: OR — if either version has it, we care about it
    merged.has_gear_data_table = p1.has_gear_data_table or p2.has_gear_data_table
    merged.has_section_views   = p1.has_section_views   or p2.has_section_views
    merged.has_detail_views    = p1.has_detail_views     or p2.has_detail_views
    merged.has_revision_table  = p1.has_revision_table   or p2.has_revision_table

    # Thresholds: take the more conservative (smaller area → catches more)
    merged.min_component_area  = min(p1.min_component_area,  p2.min_component_area)
    merged.max_component_area  = max(p1.max_component_area,  p2.max_component_area)
    merged.dim_line_min_length = min(p1.dim_line_min_length, p2.dim_line_min_length)
    merged.balloon_radius_min  = min(p1.balloon_radius_min,  p2.balloon_radius_min)
    merged.balloon_radius_max  = max(p1.balloon_radius_max,  p2.balloon_radius_max)
    merged.move_threshold_px   = min(p1.move_threshold_px,   p2.move_threshold_px)
    merged.ssim_threshold      = 0.999   # always fixed

    merged.image_width  = p1.image_width
    merged.image_height = p1.image_height
    merged.estimated_complexity = (
        "complex" if "complex" in (p1.estimated_complexity, p2.estimated_complexity)
        else "moderate" if "moderate" in (p1.estimated_complexity, p2.estimated_complexity)
        else "simple"
    )
    return merged
