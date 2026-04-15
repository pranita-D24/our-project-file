# ai_analyzer.py
# Claude Vision AI-powered drawing analysis
# Uses Anthropic API to detect semantic changes
# that pure CV cannot catch

import cv2
import numpy as np
import base64
import logging
import json
import os
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


def _encode_image_to_base64(image_array, max_size=1024):
    """
    Convert numpy image to base64 PNG for API.
    Resizes to keep API costs reasonable.
    """
    if len(image_array.shape) == 2:
        img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    else:
        img = image_array.copy()

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img   = cv2.resize(img,
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf     = BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("utf-8")


class AIAnalyzer:
    """
    Uses Claude Vision to semantically analyze
    engineering drawing changes that CV cannot catch:
    - Shape resized vs moved
    - Isometric view changes
    - Subtle geometry differences
    - Label/annotation content changes
    """

    ANALYSIS_PROMPT = """You are an expert engineering drawing reviewer.
You are comparing two versions of the same engineering drawing side by side.

The LEFT image is the ORIGINAL drawing (V1).
The RIGHT image is the MODIFIED drawing (V2).

Analyze carefully and respond ONLY with valid JSON in this exact format:
{
  "overall_verdict": "IDENTICAL|MINOR_CHANGES|SIGNIFICANT_CHANGES|MAJOR_CHANGES",
  "similarity_score": 95,
  "changes": [
    {
      "type": "MOVED|RESIZED|REMOVED|ADDED|DIMENSION_CHANGED|SHAPE_CHANGED",
      "description": "Brief description of what changed",
      "location": "top-right|top-left|bottom-right|bottom-left|center|top|bottom|left|right",
      "severity": "LOW|MEDIUM|HIGH",
      "v1_description": "What it looked like in original",
      "v2_description": "What it looks like now"
    }
  ],
  "dimension_changes": [
    {
      "original_value": "100 [3.937\"]",
      "new_value": "unknown or value if visible",
      "location": "description of where"
    }
  ],
  "ignored_elements": ["balloons", "annotations that were correctly ignored"],
  "analysis_notes": "Any important observations"
}

RULES:
1. Ignore annotation balloons (small numbered circles like ①②③)
2. Ignore revision triangles and leader lines
3. DO detect: shape resizing, shape removal, new shapes added, dimension line changes
4. DO detect: isometric view changes (corner view / 3D perspective view changes)
5. Moved objects that are IDENTICAL in shape/size should NOT be flagged
6. Only flag as RESIZED if the shape dimensions actually changed
7. Be precise about location descriptions
8. similarity_score is 0-100 where 100 = identical"""

    CHANGE_CLASSIFICATION_PROMPT = """You are analyzing a specific region of an engineering drawing.

This region shows a component that has changed between two drawing versions.
The LEFT side is the ORIGINAL, the RIGHT side is the MODIFIED version.

Determine:
1. Is this component MOVED (same shape, same size, different position)?
2. Is this component RESIZED (same shape, different dimensions)?
3. Is this component MODIFIED (shape itself changed)?
4. Is this component an annotation/balloon (should be ignored)?

Respond ONLY with JSON:
{
  "classification": "MOVED|RESIZED|MODIFIED|ANNOTATION|UNCLEAR",
  "confidence": 0.95,
  "reasoning": "brief explanation",
  "ignore": true or false
}"""

    def __init__(self):
        self.client   = None
        self.available = False
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.warning(
                    "ANTHROPIC_API_KEY not set — AI analysis disabled")
                return
            self.client    = anthropic.Anthropic(
                api_key=api_key)
            self.available = True
            logger.info("AI Analyzer (Claude Vision) ready ✅")
        except ImportError:
            logger.warning(
                "anthropic package not installed — "
                "pip install anthropic")
        except Exception as e:
            logger.error(f"AI init error: {e}")

    # ─────────────────────────────────────────
    # FULL DRAWING COMPARISON
    # ─────────────────────────────────────────
    def analyze_drawings(self, orig_gray, mod_gray):
        """
        Send both images to Claude Vision for semantic
        change analysis. Returns structured change data.
        """
        if not self.available:
            return self._fallback_result(
                "AI not available")

        try:
            # Create side-by-side comparison image
            side_by_side = self._create_side_by_side(
                orig_gray, mod_gray)

            b64 = _encode_image_to_base64(
                side_by_side, max_size=1200)

            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type"      : "base64",
                                "media_type": "image/png",
                                "data"      : b64
                            }
                        },
                        {
                            "type": "text",
                            "text": self.ANALYSIS_PROMPT
                        }
                    ]
                }]
            )

            raw = response.content[0].text.strip()
            # Strip markdown if present
            raw = raw.replace("```json", "").replace("```", "").strip()

            data = json.loads(raw)
            logger.info(
                f"AI Analysis: {data.get('overall_verdict')} "
                f"Score:{data.get('similarity_score')} "
                f"Changes:{len(data.get('changes', []))}")
            return {
                "success"  : True,
                "source"   : "claude-vision",
                "data"     : data,
                "changes"  : data.get("changes", []),
                "verdict"  : data.get("overall_verdict"),
                "ai_score" : data.get("similarity_score"),
                "notes"    : data.get("analysis_notes", "")
            }

        except json.JSONDecodeError as e:
            logger.error(f"AI JSON parse error: {e}")
            logger.error(f"Raw response: {raw[:300]}")
            return self._fallback_result(f"JSON parse error: {e}")
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return self._fallback_result(str(e))

    # ─────────────────────────────────────────
    # CLASSIFY A SPECIFIC CHANGED REGION
    # ─────────────────────────────────────────
    def classify_change_region(self, orig_gray, mod_gray,
                                bbox, padding=40):
        """
        Classify a specific bbox region — moved, resized,
        or modified. Helps disambiguate CV false positives.
        """
        if not self.available:
            return {"classification": "UNCLEAR",
                    "confidence": 0.0,
                    "ignore": False}

        try:
            x, y, w, h = bbox
            h_img, w_img = orig_gray.shape[:2]

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w_img, x + w + padding)
            y2 = min(h_img, y + h + padding)

            patch1 = orig_gray[y1:y2, x1:x2]
            patch2 = mod_gray[y1:y2, x1:x2]

            if patch1.size == 0 or patch2.size == 0:
                return {"classification": "UNCLEAR",
                        "confidence": 0.0,
                        "ignore": False}

            side_by_side = self._create_side_by_side(
                patch1, patch2, separator=4)
            b64 = _encode_image_to_base64(
                side_by_side, max_size=512)

            response = self.client.messages.create(
                model="claude-opus-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type"      : "base64",
                                "media_type": "image/png",
                                "data"      : b64
                            }
                        },
                        {
                            "type": "text",
                            "text": self.CHANGE_CLASSIFICATION_PROMPT
                        }
                    ]
                }]
            )

            raw  = response.content[0].text.strip()
            raw  = raw.replace("```json","").replace("```","").strip()
            data = json.loads(raw)
            return data

        except Exception as e:
            logger.error(f"Region classification error: {e}")
            return {"classification": "UNCLEAR",
                    "confidence": 0.0,
                    "ignore": False}

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _create_side_by_side(self, img1, img2,
                              separator=8):
        """Create a side-by-side BGR comparison image."""
        def to_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()

        b1 = to_bgr(img1)
        b2 = to_bgr(img2)

        # Make same height
        h  = max(b1.shape[0], b2.shape[0])
        w1 = b1.shape[1]
        w2 = b2.shape[1]

        canvas1 = np.ones((h, w1, 3),
                          dtype=np.uint8) * 240
        canvas2 = np.ones((h, w2, 3),
                          dtype=np.uint8) * 240
        canvas1[:b1.shape[0], :] = b1
        canvas2[:b2.shape[0], :] = b2

        # Add V1/V2 label
        cv2.putText(canvas1, "V1 - ORIGINAL",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 180), 1)
        cv2.putText(canvas2, "V2 - MODIFIED",
                    (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (180, 0, 0), 1)

        sep = np.ones((h, separator, 3),
                      dtype=np.uint8) * 80
        return np.hstack([canvas1, sep, canvas2])

    def _fallback_result(self, reason):
        return {
            "success" : False,
            "source"  : "fallback",
            "reason"  : reason,
            "changes" : [],
            "verdict" : None,
            "ai_score": None,
            "notes"   : ""
        }

    # ─────────────────────────────────────────
    # MERGE AI RESULTS WITH CV RESULTS
    # ─────────────────────────────────────────
    def merge_with_cv_results(self, cv_result,
                               ai_result):
        """
        Combines CV-detected changes with AI-detected
        changes for highest accuracy.
        """
        if not ai_result.get("success"):
            return cv_result

        ai_changes = ai_result.get("changes", [])

        # Map AI change types to our system
        type_map = {
            "MOVED"            : "matched",
            "RESIZED"          : "modified",
            "REMOVED"          : "removed",
            "ADDED"            : "added",
            "DIMENSION_CHANGED": "dimension",
            "SHAPE_CHANGED"    : "modified"
        }

        ai_insights = []
        for ch in ai_changes:
            ctype = type_map.get(
                ch.get("type", ""), "unknown")
            ai_insights.append({
                "type"       : ctype,
                "description": ch.get("description", ""),
                "location"   : ch.get("location", ""),
                "severity"   : ch.get("severity", "LOW"),
                "source"     : "ai"
            })

        # Attach AI insights to CV result
        cv_result["ai_insights"]  = ai_insights
        cv_result["ai_verdict"]   = ai_result.get("verdict")
        cv_result["ai_score"]     = ai_result.get("ai_score")
        cv_result["ai_notes"]     = ai_result.get("notes", "")
        cv_result["ai_available"] = True

        # If AI detected dimension changes, add them
        dim_changes = ai_result.get(
            "data", {}).get("dimension_changes", [])
        cv_result["ai_dimension_changes"] = dim_changes

        logger.info(
            f"AI+CV merged: {len(ai_insights)} AI insights")
        return cv_result