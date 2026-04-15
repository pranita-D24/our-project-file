# CLAUDE.md — Engineering Drawing Comparison System
## Living Architecture & Change Log

This file is updated **every time** a code change is made to the system.
It serves as a single source of truth for the current architecture, why things work the way they do, and what has changed over time.

---

## Project Overview

**Goal:** Compare two engineering drawing PDFs (V1 vs V2) and detect:
`ADDED`, `REMOVED`, `MOVED`, `RESIZED`, `CHANGED` components + `DIM_CHANGES`

**Must NOT highlight:** dimension lines, balloon circles, borders, title blocks, text notes, data tables.

**Scale:** Designed to operate across **270,000 heterogeneous drawings** from different manufacturers, standards (ISO/DIN/ANSI/JIS/BS/AS), DPI levels, and page formats — no image-specific assumptions anywhere.

---

## Current File Structure

| File | Role |
|------|------|
| `app.py` | Streamlit UI dashboard |
| `comparator.py` | Main comparison engine — orchestrates all phases |
| `pdf_reader.py` | **[NEW]** Per-drawing adaptive profile extraction |
| `layout_detector.py` | **[NEW]** Content-based layout & title block detection |
| `exclusion.py` | **[REWRITTEN]** Profile-driven exclusion mask builder |
| `segmentor.py` | SAM2 + GroundingDINO segmentation |
| `semantic_diff.py` | CLIP + DINOv2 hybrid similarity engine |
| `matcher.py` / `comparator.py` | Hungarian algorithm component matching |
| `agent_verifier.py` | Optional Claude API verification (no-op if key missing) |
| `batch_processor.py` | Batch comparison for large PDF sets |
| `pdf_processor.py` | Low-level PDF rendering |
| `profile_cache/` | **[NEW]** MD5-hashed JSON cache for DrawingProfiles |

---

## Architecture: Phases of compare()

```
compare(path1, path2)
    │
    ├─ Phase 1+2: get_or_create_profile(path1), get_or_create_profile(path2)
    │             → DrawingProfile (standard, scale, units, bboxes, thresholds)
    │             → merge_profiles(p1, p2)
    │
    ├─ Alignment: align_images_color() [SIFT homography — grayscale]
    │
    ├─ ROI: _find_drawing_roi() [uses profile.content_bbox, or projection fallback]
    │
    ├─ Phase 3: build_exclusion_mask_pair()
    │           → title block, border, balloons, dim lines — ALL from profile coords
    │
    ├─ SSIM: computed AFTER segmentation — informational only, never used as gate
    │
    ├─ Phase 5: _extract_components() [min/max area from profile.scale_ratio]
    │           compute_global_shift() + subtract_global_shift()
    │           _match_components() [Hungarian + LoFTR + shape similarity]
    │
    ├─ Dimension: _detect_dimension_lines() + _compare_dimension_lines()
    │             _ocr_dimensions() + _compare_ocr_dims() [normalize_dim_value]
    │
    ├─ Zone diff: _find_diff_zones() → pixel-level change regions
    │
    ├─ Merge: _merge_change_sources()
    │
    ├─ Layer 7: रीजनिंग (Reasoning Engine) [NEW]
    │           → Explain "WHY" a change occurred in plain English
    │
    └─ Render: _annotate() → _make_diff_heatmap() → _make_sbs()
```

---

## Key Design Rules (NEVER violate)

1. **No hardcoded geometry.** No `W * 0.82`, `H * 0.86`, `int(W * 0.90)` etc.
   - All layout boundaries come from `DrawingProfile` (content_bbox, title_block_bbox, border_bbox).
   - Fallback: projection-based content detection (sum of pixel rows/cols).

2. **No SSIM fast-exits.** SSIM is informational only.
   - Previously had `if ssim > 0.97: return early` — **this was removed** because it caused entire drawing regions to be skipped.

3. **No coordinate identity fast-exits.** `drawings_are_identical()` check was **removed**.
   - It triggered too aggressively (area_tol=0.05 matched many different drawings).

4. **Profile caching is mandatory for scale.**
   - Every PDF is profiled exactly once via MD5 hash → JSON in `profile_cache/`.
   - 270k drawings × 2 = 540k profile ops → avoids at 2–3s/profile = ~360 hours saved.

5. **SSIM threshold stays at 0.999 only** (reserved for truly pixel-perfect identical files).

6. **MOVED threshold = profile.move_threshold_px** (derived from scale, default 15px).
   - Sub-pixel rendering jitter (<15px) is absorbed by global shift correction.

---

## DrawingProfile Fields

```python
@dataclass
class DrawingProfile:
    drawing_number:    str      # e.g. "PRV73B124138"
    revision:          str      # e.g. "A", "03"
    drawing_standard:  str      # "ISO" | "ANSI" | "DIN" | "JIS" | "BS" | "AS" | "UNKNOWN"
    units:             str      # "mm" | "inch" | "dual"
    scale:             str      # "1:1", "1:2", "2:1" etc.
    scale_ratio:       float    # numeric: 1:2 → 0.5, 2:1 → 2.0

    title_block_bbox:    tuple  # (x1,y1,x2,y2) — detected location, NOT hardcoded
    border_bbox:         tuple  # actual drawing frame
    content_bbox:        tuple  # mechanical content area
    gear_data_bbox:      tuple  # gear parameter table (if present)
    revision_table_bbox: tuple  # revision history table (if present)

    has_gear_data_table: bool
    has_section_views:   bool
    has_detail_views:    bool
    has_revision_table:  bool
    estimated_complexity: str   # "simple" | "moderate" | "complex"

    min_component_area:   int   # scale-derived (base 1200 / scale²)
    max_component_area:   int   # scale-derived
    dim_line_min_length:  int   # scale-derived (base 30 × scale)
    balloon_radius_min:   int   # image-size-derived (~0.7% of height)
    balloon_radius_max:   int   # image-size-derived
    move_threshold_px:    float # scale-derived (base 15 / scale)
    ssim_threshold:       float # always 0.999 — never changes
```

---

## False Positive Logic (History of Fixes)

| What was wrong | Root cause | Fix applied |
|---|---|---|
| MOVED(2) on identical drawings | centroid_dist > 20px threshold, no offset correction | `compute_global_shift()` + subtract before matching |
| DIM_CHANGES(29) on identical drawings | OCR text "100,5" ≠ "100.5" string mismatch | `normalize_dim_value()` with regex normalization |
| GEAR DATA table treated as changed | `W * 0.82` hardcoded strip excluded it | `layout_detector.py` detects actual title block bbox |
| Identical drawings returned 100% with no analysis | SSIM > 0.97 fast-exit fired | Both fast-exits **removed** entirely |
| White blank heatmap on identical runs | Fast-exit set `diff_heatmap = np.ones * 255` | Replaced with `_make_diff_heatmap()` call |
| V1 image shown as heatmap | Fast-exit set `diff_heatmap = o_bgr.copy()` | Replaced with `_make_diff_heatmap()` call |
| Wrong thresholds on scaled drawings | Fixed min_area=1200 regardless of scale | Derived from `scale_ratio²` in `compute_adaptive_thresholds()` |

---

## Component Matching Pipeline

```
comps1 = _extract_components(og, roi, bal, min_area, max_area)
comps2 = _extract_components(mg, roi, bal, min_area, max_area)

global_dx, global_dy = compute_global_shift(comps1, comps2)  # median of top-5 anchors
if shift < move_threshold_px:
    comps2 = subtract_global_shift(comps2, dx, dy)           # correct rendering offset

_match_components(comps1, comps2, gray_v1, gray_v2)
  └─ cost_matrix[i,j] = 1 - (0.35×IoU + 0.65×LoFTR_sim)
  └─ linear_sum_assignment(cost_matrix)   # Hungarian algorithm
  └─ sim > 0.85:
       area_ratio outside (0.85, 1.15) → RESIZED
       centroid_dist > move_threshold_px → MOVED
       else → CHANGED
  └─ sim ≤ 0.85:
       → REMOVED (from V1), ADDED (from V2)
```

---

## Exclusion Mask Sources (all adaptive)

| Zone | Detection Method |
|------|----------------|
| Title block | `detect_title_block()` — density × edge-proximity × aspect ratio scoring |
| Border frame | `detect_border()` — Hough long lines (≥40% of dimension) |
| Content area | Largest region inside border minus title block |
| Balloons | HoughCircles + contour circularity — radius from profile |
| Dimension lines | HoughLinesP — min_len from profile (scale-derived) |
| Gear data table | Detected by structured table analysis (horiz+vert line crossing) |
| Revision table | Detected by structured table analysis |

---

## Changelog


### 2026-04-15 — GitHub Migration & Human Intelligence Roadmap ✅
- **CREATED** GitHub Repository: `pranita-D24/our-project-file`.
- **INITIALIZED** local Git history and successfully pushed all project files.
- **DRAFTED** Stage 7 Architecture: "Human Intelligence Agentic Layer."
- **PLANNING** migration from Coordinate-based matching to **Topological Graph Matching** to solve the "guessing" problem in complex drawings.
- **PLANNING** SAM2 + CLIP integration for semantic component identification.

### 2026-04-07 — Matching Math Fix + Ignored Dimensions on Heatmap ✅

- **MODIFIED** `comparator.py`:
  - **MOVED Component Bug Fix:** Purely moved components have an IoU of `0.0`. Under the previous formula (`0.4*IoU + 0.6*shape`), max similarity was capped at `0.60`, which meant purely moved components could never clear the `0.78` threshold and were forced into ADDED + REMOVED.
  - **New Math:** Updated matching logic to `sim = min(1.0, shape_cosine + 0.2 * iou)`. This lets perfectly identical shapes reach >0.78 on shape alone, allowing correct classification as MOVED, while using IoU to tie-break identical components.
  - **Visual Cleanup:** The heatmap now explicitly forces `smap_full[bal_full > 0] = 1.0` so balloons, dimension lines, and excluded tables do not render as differences. Yellow dimension lines have also been disabled in the heatmap overlay to fully satisfy "ignore balloons and dimensions".

### 2026-04-07 — Hotfix: Tuple of Strings Bounding Box Type Error ✅
- **MODIFIED** `comparator.py`, `exclusion.py`:
  - **Bug:** `TypeError: '>' not supported between instances of 'str' and 'int'` occurred when profile cache loaded coordinate tuples like `('121', '7'...)`.
  - **Fix:** Added `x1, y1, x2, y2 = map(int, bbox)` everywhere bounding boxes from `DrawingProfile` are unpacked (in `_find_drawing_roi`, `mask_title_block`, `mask_border`, `mask_box`).
### 2026-04-07 — RESIZED Detection + Heatmap Full-Image Fix ✅
- **MODIFIED** `comparator.py`:
  - **RESIZED threshold tightened:** Changed from `±20% (0.80-1.25)` to `±12% (0.88-1.14)`. The ±20% band was too loose — subtly resized diagrams were being classified as MOVED instead.
  - **Text filter relaxed:** Changed from `elongation > 3.5 && fill > 0.70 && area < 80k` to `elongation > 5 && fill > 0.75 && area < 50k`. The previous filter was too aggressive and was catching mechanical cross-section views (like SECTION A-A).
  - **Heatmap renders full image:** Changed from rendering heat only inside `[y1:y2, x1:x2]` (ROI) to rendering across the full image. Previously the bottom/right of the heatmap showed raw gray, making it look like only half was analyzed.

### 2026-04-07 — Layer 5 + Layer 6 Pipeline Fixes ✅
- **MODIFIED** `comparator.py`:
  - **L5 — Cost matrix rewrite:** Replaced `0.35×IoU + 0.65×LoFTR` with `0.4×IoU + 0.6×shape_cosine`. LoFTR removed — too slow (2–4s/pair) at 270k scale AND returns ~0.0 on text patches, causing text note boxes to get bad matches → false MOVED.
  - **L6 — Classification ORDER enforced (mandatory):** Changed from wrong `RESIZED→MOVED→CHANGED` to diagram-correct sequence: `1:RESIZED (area_ratio<0.80 or >1.25) → 2:MOVED (centroid_dist > move_thr) → 3:CHANGED`. Stops resized parts from being misclassified as MOVED.
  - **Match threshold:** Changed from `0.85` → `0.78` (middle ground, safer than 0.70 for heterogeneous 270k set).
  - **Text/annotation block filter in `_extract_components()`:** Added fill_ratio + elongation check — skips note boxes, section labels, wide text blocks that were registering as mechanical components and producing false MOVED/CHANGED detections.
  - **Heatmap cleanup:** Dimension line annotations suppressed below `diff_pct < 5.0%` threshold — eliminates the "Dim 771.0" clutter from rendering noise.
  - **`_match_components` signature updated:** Now takes explicit `match_thresh=0.78` and `move_threshold_px` from profile.

### 2026-04-07 — Critical: Exclusion Mask Coordinate Mismatch Fix ✅
- **MODIFIED** `comparator.py`:
  - **Bug:** `build_exclusion_mask_pair` was called with `g1c`/`g2c` (cropped ROI patches), but `DrawingProfile` bbox coordinates (`title_block_bbox`, `border_bbox` etc.) are in **full-image pixel space** from `layout_detector`. The `mask_border()` function starts all-white (255) and tries to unmask interior using full-image coords on a small crop → interior never gets unmasked → **entire mask = 255 → all components masked → 0 detections → "IDENTICAL"**.
  - **Fix:** Generate `bal_full` on full `og`/`mg` images, then crop: `bal = bal_full[y1:y2, x1:x2]`
  - **Fix:** Set `mask_border_flag=False` — the ROI crop already excludes the border region
  - **Impact:** ADDED/REMOVED/MOVED detection now works correctly

### 2026-04-07 — Adaptive Intelligence Architecture v1.0 ✅
- **CREATED** `pdf_reader.py` — DrawingProfile, read_and_profile, caching
- **CREATED** `layout_detector.py` — content-based title block, border, table detection
- **REWRITTEN** `exclusion.py` — fully profile-driven, zero hardcoded geometry
- **MODIFIED** `comparator.py`:
  - Removed SSIM fast-exit block (was `if ssim > 0.999`)
  - Removed `drawings_are_identical()` fast-exit block
  - `_find_drawing_roi()` now accepts `content_bbox` from profile
  - `compare()` calls `get_or_create_profile()` first for both files
  - `_extract_components()` uses `profile.min_component_area`, `profile.max_component_area`
  - Global shift threshold now uses `profile.move_threshold_px`

### 2026-04-07 — False Positive Fixes ✅
- **MODIFIED** `comparator.py`:
  - Added `compute_global_shift()`, `subtract_global_shift()`
  - Added `drawings_are_identical()` (later removed — too aggressive)
  - Added `normalize_dim_value()`, `fuzzy_label_match()`
  - Updated `_compare_ocr_dims()` to use normalization + 0.5% tolerance
  - Added `align_images_color()` — SIFT homography on full color images
  - Changed MOVED threshold from `> 20px` to `> 15px`

### 2026-04-07 — Core Pipeline Fixes ✅
- **MODIFIED** `comparator.py`:
  - Replaced `SAM2AutomaticMaskGenerator` with GroundingDINO-prompted `SAM2ImagePredictor`
  - Added LoFTR keypoint matching into component cost matrix
  - Added optical flow for MOVED arrow visualization
  - Integrated `agent_verifier.py` (Claude API optional, graceful no-op)
  - Added DXF fast-path via `ezdxf`
  - Fixed `NoneType` crash — fast-exit blocks now call `_make_sbs()` before returning
  - Fixed `intc` JSON serialization error — cast numpy types to native Python

---

## How to Add a Change to This File

When making any code change, append to the **Changelog** section:

```markdown
### YYYY-MM-DD — Short Description
- **CREATED/MODIFIED/DELETED** `filename.py`:
  - What changed and why
  - What bug/false-positive it fixes (if any)
```

---

## Running the System

```bash
# Start Streamlit UI
venv\Scripts\streamlit run app.py

# Run a profile test on a drawing
venv\Scripts\python.exe -c "from pdf_reader import get_or_create_profile; p = get_or_create_profile('Drawings/PRV73B124138.PDF'); print(p.drawing_standard, p.scale, p.content_bbox)"

# Clear profile cache (forces re-profiling)
rm -rf profile_cache\
```

---

## Dependencies

```
opencv-python
scikit-image
scipy
torch
kornia          # LoFTR matching
transformers    # GroundingDINO
segment-anything  # SAM2
openai-clip     # CLIP similarity
ezdxf           # DXF fast-path
fitz (PyMuPDF)  # PDF rendering + text extraction
pytesseract     # OCR fallback
anthropic       # Claude API (optional)
streamlit       # UI
```
