import cv2
import sys
import os
sys.path.insert(0, r'C:\Trivim Internship\engineering_comparison_system')
from layout_detector import detect_layout
from pdf_reader import pdf_to_image

path = r'C:\Trivim Internship\engineering_comparison_system\Drawings\PRQ93B101928 (1).pdf'
img = pdf_to_image(path)
if img is not None:
    layout = detect_layout(img)
    print(f"Title Block BBox: {layout['title_block_bbox']}")
else:
    print("Could not load image")
