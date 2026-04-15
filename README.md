# Engineering Drawing Comparison System v2.0
## Wipro Pari Private Limited

### Overview
AI-powered engineering drawing comparison using Computer Vision + Claude Vision AI.

---

### Features
| Feature | Description |
|---|---|
| ✅ **Added Detection** | New objects appearing in V2 (green) |
| ✅ **Removed Detection** | Objects missing from V2 (red) |
| ✅ **Resized Detection** | Same shape, different dimensions (orange) |
| ✅ **Dimension Changes** | Line length and value changes (yellow) |
| ✅ **Balloon Filtering** | Annotation circles/balloons ignored |
| ✅ **Moved Ignored** | Relocated objects are not flagged |
| ✅ **Claude Vision AI** | Semantic change analysis |
| ✅ **OCR Dimensions** | Reads mm/inch values from drawing |
| ✅ **Multi-format** | PDF and image (JPG/PNG) support |
| ✅ **Reports** | Visual, Heatmap, AI Panel, JSON, Text |

---

### Setup

1. **Install Python 3.10+**

2. **Install Tesseract OCR** (for dimension text reading):
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to: `C:\Program Files\Tesseract-OCR\`

3. **Install Poppler** (for PDF processing):
   - Already configured in config.py

4. **Run the application:**
   ```
   run.bat
   ```
   OR manually:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. **Get Anthropic API key** (for AI analysis):
   - Visit: https://console.anthropic.com
   - Enter key in sidebar when running

---

### File Structure
```
engineering_comparison_system/
├── app.py                  # Main Streamlit UI
├── pipeline.py             # Core orchestration
├── config.py               # All settings
├── detector.py             # Object detection
├── matcher.py              # Object matching
├── change_detector.py      # Change classification
├── aligner.py              # Image alignment
├── preprocessor.py         # Image preprocessing
├── balloon_filter.py       # Annotation filtering ← NEW
├── dimension_analyzer.py   # Dimension extraction ← NEW
├── ai_analyzer.py          # Claude Vision AI ← NEW
├── report_generator.py     # Report generation
├── pdf_processor.py        # PDF handling
├── database.py             # SQLite storage
├── requirements.txt
└── run.bat                 # Windows launcher
```

---

### What is Detected vs Ignored

**DETECTED:**
- Objects added to V2 (not in V1)
- Objects removed from V2 (was in V1)
- Objects resized (same shape, different area/dimensions)
- Dimension line length changes
- Dimension text value changes (with Tesseract)

**IGNORED (by design):**
- Moved objects (same shape + size, different position)
- Annotation balloons (numbered circles ①②)
- Revision triangles and leader lines
- Title block changes (right side excluded)

---

### Configuration (config.py)

| Setting | Default | Description |
|---|---|---|
| `MIN_CONTOUR_AREA` | 1500 | Min object size to detect |
| `SHAPE_SIMILARITY_THRESH` | 0.82 | Shape match strictness |
| `AREA_TOLERANCE` | 0.25 | Size difference allowed for matching |
| `BALLOON_MIN_RADIUS` | 12 | Min balloon circle radius |
| `BALLOON_MAX_RADIUS` | 55 | Max balloon circle radius |

---

### Reports Generated
1. **visual_report.jpg** — 3-panel comparison (Original | Modified | Differences)
2. **heatmap_report.jpg** — SSIM + pixel diff heatmaps
3. **ai_report.jpg** — AI analysis panel (if API key set)
4. **report.json** — Full structured data
5. **summary.txt** — Human-readable text report

---

### Version History
- **v2.0** — AI integration, balloon filtering, dimension analysis, production UI
- **v1.0** — Initial CV-based comparison