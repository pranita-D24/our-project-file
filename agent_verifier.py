import google.generativeai as genai
import PIL.Image, io, base64, json, os, logging, time
import cv2, numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

_KEY = os.environ.get("GEMINI_API_KEY", "")
if _KEY:
    genai.configure(api_key=_KEY)
    _FLASH = genai.GenerativeModel("gemini-1.5-flash")  # 1500/day
    _PRO   = genai.GenerativeModel("gemini-1.5-pro")    # 50/day
    _AVAILABLE = True
else:
    _AVAILABLE = False  # entire module becomes no-op silently

def _numpy_to_pil(img_bgr: np.ndarray) -> PIL.Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(img_rgb)

def calibrate_profile_with_vision(image_bgr: np.ndarray, profile):
    if not _AVAILABLE: return
    
    # Resize keeping aspect ratio to width 800
    H, W = image_bgr.shape[:2]
    scale = 800.0 / W
    small = cv2.resize(image_bgr, (800, int(H * scale)))
    pil_img = _numpy_to_pil(small)

    prompt = f"""Analyze this engineering drawing. Return ONLY valid JSON, no markdown:
{{
  "balloon_radius_range": [min_px, max_px],
  "title_block_bbox": [x1, y1, x2, y2],
  "has_isometric_view": true/false,
  "isometric_view_bboxes": [[x1,y1,x2,y2], ...],
  "dimension_text_height_px": N,
  "notes_zones": [[x1,y1,x2,y2], ...]
}}
Image width is {W}px, height is {H}px.
Balloons are small numbered circles with leader lines — measure their radius in pixels.
Title block is the table in the bottom-right corner."""

    try:
        response = _PRO.generate_content([prompt, pil_img])
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        vision = json.loads(text.strip())

        profile.balloon_radius_min = vision.get("balloon_radius_range", [12, 65])[0]
        profile.balloon_radius_max = vision.get("balloon_radius_range", [12, 65])[1]
        
        tb = vision.get("title_block_bbox")
        if tb and len(tb) == 4 and tb != [0,0,0,0]:
            profile.title_block_bbox = tuple(map(int, tb))
            
        profile._isometric_bboxes = vision.get("isometric_view_bboxes", [])
        profile._notes_zones      = vision.get("notes_zones", [])
        logger.info("Vision calibration successful.")
    except Exception as e:
        logger.warning(f"Vision calibration failed: {e}")

def verify_match(v1_crop_bgr: np.ndarray, v2_crop_bgr: np.ndarray, proposed_label: str, conf: float) -> dict:
    if not _AVAILABLE:
        return {"label": proposed_label, "confidence": conf, "reason": "no-op"}
    
    if v1_crop_bgr is None or v2_crop_bgr is None or v1_crop_bgr.size == 0 or v2_crop_bgr.size == 0:
        return {"label": proposed_label, "confidence": conf, "reason": "empty-crop"}

    pil1 = _numpy_to_pil(v1_crop_bgr)
    pil2 = _numpy_to_pil(v2_crop_bgr)

    prompt = f"""First image = V1 component. Second image = V2 component.
System proposed label: {proposed_label} (confidence {conf:.2f})

Choose the correct label:
- MOVED     : identical geometry, different position
- ADDED     : exists only in V2
- REMOVED   : exists only in V1
- CHANGED   : same position, geometry modified
- RESIZED   : same shape, different scale
- FALSE_POSITIVE : this is a balloon, dimension line, annotation, border — not a real component

Return ONLY valid JSON:
{{"label": "...", "confidence": 0.0-1.0, "reason": "one sentence"}}"""

    try:
        time.sleep(0.5)
        response = _FLASH.generate_content([prompt, pil1, pil2])
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        result = json.loads(text.strip())
        return result
    except Exception as e:
        logger.warning(f"verify_match failed: {e}")
        return {"label": proposed_label, "confidence": conf, "reason": "error"}

def _centroid_inside(change, bbox):
    cx, cy = change.get("centroid_x", 0), change.get("centroid_y", 0)
    x1, y1, x2, y2 = bbox
    return x1 <= cx <= x2 and y1 <= cy <= y2

def self_correct_output(annotated_bgr: np.ndarray, change_list: list) -> list:
    if not _AVAILABLE: return change_list

    # Resize preview keeping aspect ratio
    H, W = annotated_bgr.shape[:2]
    scale = 1000.0 / max(W, H)
    small = cv2.resize(annotated_bgr, (int(W * scale), int(H * scale)))
    pil_img = _numpy_to_pil(small)

    prompt = """This is an engineering drawing comparison output.
Colored highlight boxes show detected changes.
Identify ALL false positive regions — balloons, dimension lines,
text annotations, title block cells, border lines that were
incorrectly flagged as component changes.

Return ONLY valid JSON:
{
  "false_positive_bboxes": [[x1,y1,x2,y2], ...],
  "genuine_change_count": N,
  "summary": "one sentence"
}"""

    try:
        response = _PRO.generate_content([prompt, pil_img])
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        result = json.loads(text.strip())

        for bbox in result.get("false_positive_bboxes", []):
            # Gemini returns bboxes in resized image space [0-1000 range usually, or relative to the 1000px max side]
            # If Gemini uses 0-1000 normalized coords, we'd need to scale by W/1000, H/1000.
            # But the prompt implies pixel coords in the image it sees.
            # If Gemini sees a 1000px max-side image, we scale back:
            x1, y1, x2, y2 = [int(v / scale) for v in bbox]
            change_list = [c for c in change_list if not _centroid_inside(c, [x1, y1, x2, y2])]
            
        logger.info(f"Self-correction summary: {result.get('summary')}")
        return change_list
    except Exception as e:
        logger.warning(f"self_correct_output failed: {e}")
        return change_list

def verify_changes(raw_counts: dict, path1: str, path2: str) -> dict:
    """
    Sanity check comparison counts using Gemini.
    Returns verified/corrected counts.
    """
    if not _AVAILABLE:
        return raw_counts
        
    try:
        # Load small thumbnails for context
        img1 = pdf_to_image(path1, dpi=72)
        img2 = pdf_to_image(path2, dpi=72)
        if img1 is None or img2 is None:
            return raw_counts
            
        pil1 = _numpy_to_pil(img1)
        pil2 = _numpy_to_pil(img2)
        
        prompt = f"""These are two versions of an engineering drawing.
Computer vision detected these changes: {raw_counts}
Does this scale of change look correct based on the visual difference?
Focus ONLY on major additions or removals.
Return ONLY valid JSON:
{{
  "added": N, "removed": N, "resized": N, "changed": N, "moved": N,
  "confidence": 0.0-1.0,
  "ai_note": "one sentence"
}}"""
        response = _FLASH.generate_content([prompt, pil1, pil2])
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        verified = json.loads(text.strip())
        
        # Merge AI results back (conservative)
        final = raw_counts.copy()
        for k in ["added", "removed", "resized", "changed", "moved"]:
            if k in verified:
                final[k] = verified[k]
        
        logger.info(f"AI Verification completed: {verified.get('ai_note')}")
        return final
    except Exception as e:
        logger.warning(f"verify_changes failed: {e}")
        return raw_counts
