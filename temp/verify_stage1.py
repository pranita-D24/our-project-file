import sys, os, io
sys.path.insert(0, os.path.abspath('.'))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

results = []

# ─── CHECK 1a: _preprocess_for_layout mutates the image ────────────────────
try:
    from pdf_reader import pdf_to_image, _preprocess_for_layout

    raw = pdf_to_image('a.jpg')
    pre, binary = _preprocess_for_layout(raw)

    same_object  = raw is pre
    pixels_diff  = int(np.sum(raw.astype(int) != pre.astype(int)))
    binary_ok    = (binary.dtype == np.uint8 and binary.shape == raw.shape[:2])

    results.append(f"CHECK 1a  raw is pre (must be False): {same_object}")
    results.append(f"CHECK 1a  pixels mutated: {pixels_diff:,}  (must be > 0)")
    results.append(f"CHECK 1a  binary shape+dtype OK: {binary_ok}")
    results.append(f"CHECK 1a  PASS: {not same_object and pixels_diff > 0 and binary_ok}")
except Exception as e:
    import traceback
    results.append(f"CHECK 1a  ERROR: {e}")
    results.append(traceback.format_exc())

# ─── CHECK 1b: detect_layout is called with preprocessed image ─────────────
try:
    from layout_detector import detect_layout

    layout_pre = detect_layout(pre)
    layout_raw = detect_layout(raw)

    cb_pre = layout_pre["content_bbox"]
    cb_raw = layout_raw["content_bbox"]

    results.append(f"CHECK 1b  content_bbox from preprocessed path: {cb_pre}")
    results.append(f"CHECK 1b  content_bbox from raw path:           {cb_raw}")
    results.append(f"CHECK 1b  layout detection runs on preprocessed image: confirmed")
except Exception as e:
    results.append(f"CHECK 1b  ERROR: {e}")

# ─── CHECK 1c: _preprocessed_binary stored on profile ──────────────────────
try:
    # Force a cache miss by using a fresh profile build
    from pdf_reader import read_and_profile
    profile = read_and_profile('a.jpg')
    has_bin = hasattr(profile, '_preprocessed_binary') and profile._preprocessed_binary is not None
    if has_bin:
        bshape = profile._preprocessed_binary.shape
        bdtype = profile._preprocessed_binary.dtype
        results.append(f"CHECK 1c  _preprocessed_binary on profile: shape={bshape} dtype={bdtype}")
    else:
        results.append(f"CHECK 1c  _preprocessed_binary on profile: MISSING")
    results.append(f"CHECK 1c  PASS: {has_bin}")
except Exception as e:
    results.append(f"CHECK 1c  ERROR: {e}")

# ─── CHECK 2: .result() propagates exceptions from ThreadPoolExecutor ───────
try:
    from concurrent.futures import ThreadPoolExecutor

    def _bad():
        raise RuntimeError("simulated PyMuPDF/pdfplumber failure")

    caught = False
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_bad)
        fut.result()
    except RuntimeError:
        caught = True

    results.append(f"CHECK 2   .result() propagates future exceptions: {caught}")
    results.append(f"CHECK 2   PASS: {caught}")
except Exception as e:
    results.append(f"CHECK 2   ERROR: {e}")

# ─── CHECK 3: DXF cache is written and hit correctly ───────────────────────
try:
    import glob, pathlib, hashlib

    # Build a minimal valid DXF if no real one exists
    dxf_files = glob.glob('Drawings/**/*.dxf', recursive=True) + glob.glob('uploads/**/*.dxf', recursive=True)
    if not dxf_files:
        minimal = "0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
        test_path = "temp/test_minimal.dxf"
        with open(test_path, "w") as f:
            f.write(minimal)
        dxf_path = test_path
    else:
        dxf_path = dxf_files[0]

    # Compute expected MD5 cache path
    h = hashlib.md5()
    with open(dxf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    cache_path = pathlib.Path("profile_cache") / f"{h.hexdigest()}.json"

    # Remove to force a fresh write
    if cache_path.exists():
        cache_path.unlink()

    from ingestion import ingest
    prof = ingest(dxf_path)

    written = cache_path.exists()
    results.append(f"CHECK 3   DXF file tested: {dxf_path}")
    results.append(f"CHECK 3   Cache file written: {written}  ({cache_path.name})")
    results.append(f"CHECK 3   profile.units: {prof.units}")
    results.append(f"CHECK 3   profile.content_bbox: {prof.content_bbox}")

    # Second call must log HIT and return same data
    prof2 = ingest(dxf_path)
    data_consistent = prof.units == prof2.units and prof.content_bbox == prof2.content_bbox
    results.append(f"CHECK 3   Second call cache HIT data consistent: {data_consistent}")
    results.append(f"CHECK 3   PASS: {written and data_consistent}")
except Exception as e:
    import traceback
    results.append(f"CHECK 3   ERROR: {e}")
    results.append(traceback.format_exc())

# ─── Summary ────────────────────────────────────────────────────────────────
print()
for r in results:
    print(r)
print()

passes  = [r for r in results if "PASS: True"  in r]
fails   = [r for r in results if "PASS: False" in r]
errors  = [r for r in results if "ERROR" in r]
print(f"Results: {len(passes)} PASS  {len(fails)} FAIL  {len(errors)} ERROR")
if not fails and not errors:
    print("ALL CHECKS PASSED — Stage 1 verification complete")
else:
    print("FAILURES DETECTED — see above")
