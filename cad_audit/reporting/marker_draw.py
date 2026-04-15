import cv2
import numpy as np
from PIL import ImageDraw, ImageFont, Image

def draw_markers(image_array, added_spans, removed_spans):
    """Draws precise forensic markers using PIL for high-fidelity Unicode support."""
    # Convert BGR (OpenCV) to RGB (PIL)
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    
    # Optional: Load a default font if available, or use basic
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for span in added_spans:
        cx, cy = int(span["centroid"][0]), int(span["centroid"][1])
        # Green circles for additions
        draw.ellipse([cx-12, cy-12, cx+12, cy+12], 
                     outline=(0,200,0), width=3)
        draw.text((cx+15, cy-8), span["text"], 
                  fill=(0,200,0), font=font)
    
    for span in removed_spans:
        cx, cy = int(span["centroid"][0]), int(span["centroid"][1])
        # Red circles for removals
        draw.ellipse([cx-12, cy-12, cx+12, cy+12], 
                     outline=(200,0,0), width=3)
        draw.text((cx+15, cy-8), span["text"], 
                  fill=(200,0,0), font=font)
    
    # Convert back to BGR for OpenCV compatibility
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
