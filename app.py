# app.py — Engineering Drawing Comparison System v5.0
# Streamlit UI. Drop 2 files → instant smart diff.
# Uses comparator.py for all CV logic.

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json
from datetime import datetime

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

st.set_page_config(
    page_title="DrawingDiff v5",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded")

# ── Dark theme CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
.stApp,.main{ background:#0d1117; color:#c9d1d9; }
[data-testid="stSidebar"]{ background:#010409; border-right:1px solid #21262d; }

.hdr{ background:#0d1117; border:1px solid #21262d;
  border-left:4px solid #58a6ff; padding:18px 24px;
  border-radius:6px; margin-bottom:16px; }
.hdr h1{ font-size:1.4rem; font-weight:700; color:#f0f6fc; margin:0 0 4px; }
.hdr p { font-family:'JetBrains Mono',monospace; font-size:0.68rem;
  color:#58a6ff; letter-spacing:1.5px; margin:0; }

.mc{ background:#161b22; border:1px solid #21262d; border-radius:8px;
  padding:14px; text-align:center; }
.mv{ font-size:2rem; font-weight:700; line-height:1; margin-bottom:2px; }
.ml{ font-family:'JetBrains Mono',monospace; font-size:0.58rem;
  text-transform:uppercase; letter-spacing:1.5px; color:#6e7681; }

.v-i{ background:#0f2d0f; border:1px solid #3fb950; color:#3fb950;
  padding:10px 18px; border-radius:6px; font-weight:600;
  font-family:'JetBrains Mono',monospace; text-align:center; }
.v-n{ background:#0c2d38; border:1px solid #39c5cf; color:#39c5cf;
  padding:10px 18px; border-radius:6px; font-weight:600;
  font-family:'JetBrains Mono',monospace; text-align:center; }
.v-m{ background:#2d2200; border:1px solid #d29922; color:#d29922;
  padding:10px 18px; border-radius:6px; font-weight:600;
  font-family:'JetBrains Mono',monospace; text-align:center; }
.v-j{ background:#2d0f0f; border:1px solid #f85149; color:#f85149;
  padding:10px 18px; border-radius:6px; font-weight:600;
  font-family:'JetBrains Mono',monospace; text-align:center; }

.cr{ background:#161b22; border:1px solid #21262d; border-radius:6px;
  padding:10px 14px; margin:5px 0; font-size:0.82rem; }
.badge{ display:inline-block; padding:2px 9px; border-radius:10px;
  font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:600;
  margin-right:6px; }
.ba{ background:#0f2d0f; color:#3fb950; border:1px solid #3fb950; }
.br{ background:#2d0f0f; color:#f85149; border:1px solid #f85149; }
.brs{ background:#1a1400; color:#d29922; border:1px solid #d29922; }
.bc{ background:#0c2033; color:#39c5cf; border:1px solid #39c5cf; }
.bm{ background:#1a1a00; color:#c8c800; border:1px solid #c8c800; }
.bd{ background:#0e1a1a; color:#00d4d4; border:1px solid #00d4d4; }
.ib{ background:#161b22; border-left:3px solid #58a6ff; padding:8px 12px;
  border-radius:0 6px 6px 0; font-size:0.78rem; color:#8b949e; margin:6px 0; }

.stButton>button{ background:linear-gradient(135deg,#1f6feb,#58a6ff);
  color:#fff; border:none; border-radius:6px; padding:10px 24px;
  font-family:'JetBrains Mono',monospace; font-size:0.8rem; font-weight:600;
  width:100%; }
.stTabs [data-baseweb="tab-list"]{ background:#161b22;
  border-radius:8px 8px 0 0; gap:2px; padding:4px; }
.stTabs [data-baseweb="tab"]{ font-family:'JetBrains Mono',monospace;
  font-size:0.72rem; color:#6e7681; }
.stTabs [aria-selected="true"]{ color:#58a6ff; background:#0d1117; border-radius:6px; }
#MainMenu,footer,header{ visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load comparator ──
try:
    from comparator import compare, CompareResult
    _loaded = True
except ImportError as e:
    _loaded = False
    _load_err = str(e)


def bgr_rgb(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_tmp(f, ext):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    t.write(f.read())
    t.close()
    return t.name


def get_img_path(f, ext):
    if ext != ".pdf":
        return save_tmp(f, ext)
    try:
        import fitz
        p   = save_tmp(f, ".pdf")
        doc = fitz.open(p)
        px  = doc[0].get_pixmap(
            matrix=fitz.Matrix(220/72, 220/72))
        out = p.replace(".pdf", ".jpg")
        px.save(out)
        return out
    except ImportError:
        st.error("Install PyMuPDF: pip install pymupdf")
        st.stop()


def badge(ct):
    cls = {"ADDED":"ba","REMOVED":"br","RESIZED":"brs",
           "CHANGED":"bc","MOVED":"bm"}.get(ct,"bc")
    return f'<span class="badge {cls}">{ct}</span>'


# ── Header ──
st.markdown("""
<div class="hdr">
  <p>WIPRO PARI PRIVATE LIMITED · v5.0 · SMART DIFF ENGINE</p>
  <h1>⚙ Engineering Drawing Comparison</h1>
</div>""", unsafe_allow_html=True)

if not _loaded:
    st.error(f"comparator.py error: {_load_err}")
    st.stop()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙ Settings")
    sensitivity = st.slider(
        "Detection Sensitivity", 0.0, 1.0, 0.55, 0.05,
        help="Higher = catches smaller changes")

    st.markdown("""
    <div class="ib"><b>What it detects:</b><br>
    🟢 <b>ADDED</b> — new content in V2<br>
    🔴 <b>REMOVED</b> — content missing in V2<br>
    🟠 <b>RESIZED</b> — same component, different size<br>
    🔵 <b>CHANGED</b> — content modified in place<br>
    🟡 <b>MOVED</b> — relocated (shown, not flagged)<br><br>
    ⬜ <b>Ignored:</b> Balloons, annotations,<br>
    &nbsp;&nbsp;&nbsp;title block, revision markers
    </div>""", unsafe_allow_html=True)

    st.divider()

    if os.environ.get("CLAUDE_API_KEY"):
        st.sidebar.success("AI verification: ON")
    else:
        st.sidebar.info("AI verification: OFF (set CLAUDE_API_KEY to enable)")

# ── Upload ──
st.markdown("#### Upload Drawings")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        '<p style="color:#58a6ff;font-size:0.75rem;'
        'font-family:monospace;margin-bottom:4px;">'
        'V1 — ORIGINAL</p>',
        unsafe_allow_html=True)
    f1 = st.file_uploader(
        "V1", type=["pdf", "dxf", "dwg", "jpg", "jpeg", "png"],
        key="f1", label_visibility="collapsed")
with c2:
    st.markdown(
        '<p style="color:#f85149;font-size:0.75rem;'
        'font-family:monospace;margin-bottom:4px;">'
        'V2 — MODIFIED</p>',
        unsafe_allow_html=True)
    f2 = st.file_uploader(
        "V2", type=["pdf", "dxf", "dwg", "jpg", "jpeg", "png"],
        key="f2", label_visibility="collapsed")

if f1 and f2:
    ext1 = os.path.splitext(f1.name.lower())[1]
    ext2 = os.path.splitext(f2.name.lower())[1]

    if ext1 in [".jpg",".jpeg",".png"] and \
       ext2 in [".jpg",".jpeg",".png"]:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.image(f1, caption="V1 Preview",
                     use_container_width=True)
        with pc2:
            st.image(f2, caption="V2 Preview",
                     use_container_width=True)

    st.markdown("---")
    _, bc, _ = st.columns([1,2,1])
    with bc:
        run = st.button(
            "▶  COMPARE DRAWINGS",
            type="primary",
            use_container_width=True)

    if run:
        prog = st.progress(0)
        stat = st.empty()
        import time
        t_start = time.time()

        try:
            stat.text("Saving files...")
            prog.progress(10)
            p1 = get_img_path(f1, ext1)
            p2 = get_img_path(f2, ext2)

            stat.text("Aligning drawings (SIFT → ORB → Phase)...")
            prog.progress(25)

            stat.text("Detecting components + balloons...")
            prog.progress(45)

            stat.text("Matching components (position-independent)...")
            prog.progress(60)

            stat.text("Analyzing dimension lines + OCR...")
            prog.progress(75)

            if ext1 in [".dxf", ".dwg"] and ext2 in [".dxf", ".dwg"]:
                st.info("DXF source detected — using fast entity diff, skipping vision pipeline")
                from dxf_parser import diff_dxf
                result = diff_dxf(p1, p2)
            else:
                result: CompareResult = compare(
                    p1, p2, sensitivity=sensitivity)

            prog.progress(92)
            stat.text("Building reports...")

            # Old AIAnalyzer removed since agent_verifier runs inside the core pipeline.
            ai_data = None

            prog.progress(100)
            stat.empty()
            prog.empty()

            t_end = time.time()
            result.processing_info["time_sec"] = \
                round(t_end - t_start, 2)

            st.session_state["result"]  = result
            st.session_state["ai_data"] = ai_data
            st.session_state["names"]   = (f1.name, f2.name)

        except Exception as e:
            prog.empty()
            stat.empty()
            st.error(f"Error: {e}")
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════
if "result" in st.session_state:
    res: CompareResult = st.session_state["result"]
    ai_data            = st.session_state.get("ai_data")
    names              = st.session_state.get("names",("",""))

    st.markdown("---")

    # ── Verdict banner ──
    vmap = {
        "IDENTICAL / VERY SIMILAR": "v-i",
        "MINOR CHANGES"           : "v-n",
        "MODERATE CHANGES"        : "v-m",
        "MAJOR CHANGES"           : "v-j",
    }
    vc = vmap.get(res.verdict, "v-j")
    st.markdown(
        f'<div class="{vc}">  {res.verdict}'
        f'  &nbsp;|&nbsp;  Similarity: {res.similarity}%'
        f'  &nbsp;|&nbsp;  '
        f'Time: {res.processing_info.get("time_sec","?")}s'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics ──
    info = res.processing_info
    an   = info.get("added", 0)
    rn   = info.get("removed", 0)
    rsn  = info.get("resized", 0)
    cn   = info.get("changed", 0)
    mn   = info.get("moved", 0)
    dn   = info.get("dim_changes", 0)

    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.markdown(f'<div class="mc"><div class="mv" style="color:#58a6ff">{res.similarity}%</div><div class="ml">Similarity</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="mc"><div class="mv" style="color:#3fb950">{an}</div><div class="ml">Added</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="mc"><div class="mv" style="color:#f85149">{rn}</div><div class="ml">Removed</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="mc"><div class="mv" style="color:#d29922">{rsn}</div><div class="ml">Resized</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="mc"><div class="mv" style="color:#39c5cf">{cn}</div><div class="ml">Changed</div></div>', unsafe_allow_html=True)
    m6.markdown(f'<div class="mc"><div class="mv" style="color:#00d4d4">{dn}</div><div class="ml">Dim Changes</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs([
        "📐 Visual Diff",
        "🌡 Heatmap",
        "🤖 AI Insights",
        "📏 Dimensions",
        "📋 Change List",
        "⬇ Export"
    ])

    # ── TAB 1: Visual Diff ──
    with tabs[0]:
        a, b, c = st.columns(3)
        with a:
            st.caption("ORIGINAL — V1")
            st.image(bgr_rgb(res.orig_annotated),
                     use_container_width=True)
        with b:
            st.caption("MODIFIED — V2")
            st.image(bgr_rgb(res.mod_annotated),
                     use_container_width=True)
        with c:
            st.caption("DIFF HEATMAP")
            st.image(bgr_rgb(res.diff_heatmap),
                     use_container_width=True)

        st.markdown("---")
        st.caption("Full side-by-side report")
        st.image(bgr_rgb(res.side_by_side),
                 use_container_width=True)

    # ── TAB 2: Heatmap ──
    with tabs[1]:
        st.markdown("""
        <div class="ib">
        Blue = high similarity &nbsp;·&nbsp;
        Red = high difference.<br>
        Yellow lines = dimension changes detected.
        Boxes = change zones.
        </div>""", unsafe_allow_html=True)
        st.image(bgr_rgb(res.diff_heatmap),
                 use_container_width=True)
        st.progress(int(res.similarity),
                    text=f"Similarity: {res.similarity}%")
        st.metric("SSIM Score",
                  f"{round(res.overall_ssim*100,2)}%")
        st.metric("Pixel Diff",
                  f"{round(res.pixel_diff_pct,2)}%")

    # ── TAB 3: AI Insights ──
    with tabs[2]:
        if not ai_data or not ai_data.get("success"):
            st.markdown("""
            <div class="ib">
            <b>AI analysis not active.</b><br>
            Enter your Anthropic API key in the sidebar
            to enable Claude Vision semantic analysis.
            </div>""", unsafe_allow_html=True)
        else:
            changes = ai_data.get("changes", [])
            ac1, ac2 = st.columns([2,1])
            with ac2:
                st.metric("AI Score",
                          f"{ai_data.get('ai_score','?')}%")
                st.metric("AI Verdict",
                          ai_data.get("verdict","N/A"))
                st.metric("AI Changes",
                          len(changes))
            with ac1:
                for ch in changes:
                    sev  = ch.get("severity","LOW")
                    icon = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢"}.get(sev,"⚪")
                    with st.expander(
                            f"{icon} {ch.get('type','?')} — "
                            f"{ch.get('location','')}"):
                        st.write(
                            f"**Description:** "
                            f"{ch.get('description','')}")
                        if ch.get("v1_description"):
                            st.write(
                                f"**Original:** "
                                f"{ch['v1_description']}")
                        if ch.get("v2_description"):
                            st.write(
                                f"**Modified:** "
                                f"{ch['v2_description']}")

    # ── TAB 4: Dimensions ──
    with tabs[3]:
        st.markdown(
            '<span style="font-family:monospace;font-size:0.7rem;'
            'color:#00d4d4;text-transform:uppercase;'
            'letter-spacing:1px;">📏 DIMENSION ANALYSIS</span>',
            unsafe_allow_html=True)

        if not res.dim_changes:
            st.success("No dimension changes detected.")
        else:
            for i, ch in enumerate(res.dim_changes, 1):
                ctype = ch.get("type", "LINE")
                if ctype == "OCR":
                    st.markdown(
                        f'<div class="cr">'
                        f'<span class="badge bd">OCR VALUE</span>'
                        f' <b style="color:#f0f6fc">'
                        f'{ch.get("v1_text","?")} → '
                        f'{ch.get("v2_text","?")}</b>'
                        f'<span style="color:#8b949e"> | '
                        f'Δ {ch.get("delta",0):+.1f} mm</span>'
                        f'</div>',
                        unsafe_allow_html=True)
                else:
                    orient = ch.get("orient","?")
                    pct    = ch.get("diff_pct",0)
                    v1l    = ch.get("v1_len",0)
                    v2l    = ch.get("v2_len",0)
                    st.markdown(
                        f'<div class="cr">'
                        f'<span class="badge bd">LINE [{orient}]</span>'
                        f' <b style="color:#f0f6fc">'
                        f'{v1l:.0f}px → {v2l:.0f}px</b>'
                        f'<span style="color:#8b949e"> | '
                        f'Δ{pct:.1f}%  '
                        f'({ch.get("diff_px",0):.0f}px)</span>'
                        f'</div>',
                        unsafe_allow_html=True)

    # ── TAB 5: Change List ──
    with tabs[4]:
        if not res.regions:
            st.success("No changes detected.")
        else:
            for i, r in enumerate(res.regions, 1):
                if r.change_type == "MOVED":
                    continue   # moved objects — listed but greyed out
                st.markdown(
                    f'<div class="cr">'
                    f'<b style="color:#6e7681;">#{i}</b> &nbsp;'
                    f'{badge(r.change_type)}'
                    f'<b style="color:#f0f6fc">{r.label}</b> &nbsp;'
                    f'<span style="color:#8b949e">'
                    f'| {r.changed_pct:.1f}% changed '
                    f'| SSIM {r.local_ssim:.3f} '
                    f'| {r.w}×{r.h}px'
                    f'{" | " + r.detail if r.detail else ""}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True)

            if mn > 0:
                st.markdown(
                    f'<div class="cr" style="opacity:0.5">'
                    f'{badge("MOVED")}'
                    f'<span style="color:#6e7681">'
                    f'{mn} moved objects detected and ignored</span>'
                    f'</div>',
                    unsafe_allow_html=True)

        st.markdown("---")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("V1 Components",
                   info.get("components_v1",0))
        sc2.metric("V2 Components",
                   info.get("components_v2",0))
        sc3.metric("Alignment",
                   info.get("alignment","?"))
        sc4.metric("Processing",
                   f"{info.get('time_sec','?')}s")

        # JSON data
        with st.expander("Raw JSON data"):
            jdata = {
                "verdict"      : res.verdict,
                "similarity"   : res.similarity,
                "ssim"         : round(res.overall_ssim, 4),
                "pixel_diff_pct": round(res.pixel_diff_pct, 3),
                "v1"           : names[0],
                "v2"           : names[1],
                "processing"   : res.processing_info,
                "regions"      : [
                    {"type"       : r.change_type,
                     "zone"       : r.label,
                     "changed_pct": round(r.changed_pct, 2),
                     "bbox"       : [r.x, r.y, r.w, r.h],
                     "detail"     : r.detail}
                    for r in res.regions],
                "dimension_changes": res.dim_changes,
                "ai_available" : bool(
                    ai_data and ai_data.get("success")),
                "generated"    : datetime.now().isoformat()
            }
            st.json(jdata)

    # ── TAB 6: Export ──
    with tabs[5]:
        def enc(img, q=88):
            if img is None:
                return b""
            ok, buf = cv2.imencode(
                ".jpg", img,
                [cv2.IMWRITE_JPEG_QUALITY, q])
            return buf.tobytes() if ok else b""

        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "⬇ Side-by-side (JPG)",
                enc(res.side_by_side),
                "comparison.jpg", "image/jpeg",
                use_container_width=True)
            st.download_button(
                "⬇ Heatmap (JPG)",
                enc(res.diff_heatmap),
                "heatmap.jpg", "image/jpeg",
                use_container_width=True)
        with e2:
            st.download_button(
                "⬇ Annotated V1 (JPG)",
                enc(res.orig_annotated),
                "annotated_v1.jpg", "image/jpeg",
                use_container_width=True)
            st.download_button(
                "⬇ Annotated V2 (JPG)",
                enc(res.mod_annotated),
                "annotated_v2.jpg", "image/jpeg",
                use_container_width=True)
        with e3:
            jdata = {
                "verdict"           : res.verdict,
                "similarity"        : res.similarity,
                "processing"        : res.processing_info,
                "regions"           : [
                    {"type": r.change_type,
                     "zone": r.label,
                     "changed_pct": round(r.changed_pct, 2),
                     "bbox": [r.x,r.y,r.w,r.h]}
                    for r in res.regions],
                "dimension_changes" : res.dim_changes,
                "v1"                : names[0],
                "v2"                : names[1],
                "generated"         : datetime.now().isoformat()
            }
            st.download_button(
                "⬇ JSON Report",
                json.dumps(jdata, indent=2, cls=NpEncoder).encode(),
                "report.json", "application/json",
                use_container_width=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;color:#30363d">
      <div style="font-size:3.5rem">⚙</div>
      <div style="font-size:1rem;color:#58a6ff;
      font-weight:700;margin:12px 0 6px;">Ready to Compare</div>
      <div style="font-size:0.82rem;color:#6e7681;
      font-family:monospace;">
        Upload two drawing files above and click
        COMPARE DRAWINGS
      </div>
    </div>""", unsafe_allow_html=True)
