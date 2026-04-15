# config.py — Central Configuration v2.1
import os

# ══════════════════════════════════════
# DIRECTORIES
# ══════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
IMAGE_DIR    = os.path.join(BASE_DIR, "images")
VERSION_DIR  = os.path.join(BASE_DIR, "versions")
REPORT_DIR   = os.path.join(BASE_DIR, "reports")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
TEMP_DIR     = os.path.join(BASE_DIR, "temp")

for folder in [UPLOAD_DIR, IMAGE_DIR, VERSION_DIR,
               REPORT_DIR, DATABASE_DIR, TEMP_DIR]:
    os.makedirs(folder, exist_ok=True)

# ══════════════════════════════════════
# IMAGE SETTINGS
# ══════════════════════════════════════
IMAGE_DPI     = 220
TARGET_WIDTH  = 1920
TARGET_HEIGHT = 1440
TARGET_SIZE   = (TARGET_WIDTH, TARGET_HEIGHT)

# ══════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════
CONTRAST_FACTOR = 1.5
NOISE_KERNEL    = (3, 3)
THRESHOLD_VALUE = 25

# ══════════════════════════════════════
# ROI — Drawing Area Only
# Excludes title block from comparison
# Right ~20% = title block (drawing no, rev, dates)
# Bottom ~22% = notes and title block
# ══════════════════════════════════════
ROI_X1_PCT = 0.01   # left start
ROI_X2_PCT = 0.80   # right end  (exclude title block)
ROI_Y1_PCT = 0.01   # top start
ROI_Y2_PCT = 0.78   # bottom end (exclude notes block)

# ══════════════════════════════════════
# OBJECT DETECTION
# ══════════════════════════════════════
MIN_CONTOUR_AREA  = 800
MAX_CONTOUR_AREA  = 900000
ORB_FEATURES      = 5000
GOOD_MATCH_RATIO  = 0.25

# ══════════════════════════════════════
# OBJECT MATCHING
# ══════════════════════════════════════
AREA_TOLERANCE          = 0.35
SHAPE_SIMILARITY_THRESH = 0.78
POSITION_IGNORE         = True

# ══════════════════════════════════════
# CHANGE DETECTION THRESHOLDS
# ══════════════════════════════════════
VERY_SIMILAR_THRESHOLD = 95
MODERATE_THRESHOLD     = 80

# ══════════════════════════════════════
# BALLOON DETECTION
# FIX: Lowered circularity to 0.72
# so imperfect callout circles with
# leader lines are correctly filtered
# Raised min radius to 14 to skip
# tiny line intersection artifacts
# ══════════════════════════════════════
BALLOON_MIN_RADIUS  = 14
BALLOON_MAX_RADIUS  = 55
BALLOON_CIRCULARITY = 0.72

# ══════════════════════════════════════
# DIMENSION DETECTION
# FIX: Raised DIM_LINE_MIN_LENGTH 30→70
# At 30px every scan artifact and short
# hatch line was counted as dimension
# Engineering dim lines are 50-80px min
# ══════════════════════════════════════
DIM_LINE_MIN_LENGTH = 70
DIM_TEXT_MARGIN     = 45

# ══════════════════════════════════════
# AI SETTINGS
# ══════════════════════════════════════
AI_MODEL    = "claude-sonnet-4-5"
AI_ENABLED  = True

# ══════════════════════════════════════
# REPORT SETTINGS
# ══════════════════════════════════════
REPORT_FORMAT = "jpg"
REPORT_DPI    = 150

# ══════════════════════════════════════
# SYSTEM INFO
# ══════════════════════════════════════
SYSTEM_NAME = "Engineering Drawing Comparison System"
VERSION     = "2.1.0"
COMPANY     = "Wipro Pari Private Limited"
AUTHOR      = "Trivim Internship"

# ══════════════════════════════════════
# TOOLS
# ══════════════════════════════════════
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH  = r"C:\Users\Pranita\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# ══════════════════════════════════════
# VERSION SETTINGS
# ══════════════════════════════════════
VERSION_PREFIX = "v"

# ══════════════════════════════════════
# OCR SETTINGS
# ══════════════════════════════════════
OCR_LANGUAGE = "eng"
OCR_CONFIG   = "--psm 6"

print(f"{SYSTEM_NAME} v{VERSION}")
print(f"Company  : {COMPANY}")
print("Folders  : Ready [OK]")