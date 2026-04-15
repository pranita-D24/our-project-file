import fitz
import sys
import os

from stage4_geometry import Stage4Engine

doc1 = fitz.open('Drawings/PRV73B124139.PDF')
page = doc1[0]
engine = Stage4Engine()

drawings = page.get_drawings()
print(f"Total parsed drawings: {len(drawings)}")

# Find everything that is dropped by normalize_path
for i, d in enumerate(drawings):
    r = d.get('rect')
    if not r: continue
    
    # 35-40% from left, 50% from top
    # Let's see if centroid matches and it gets dropped
    cx, cy = (r[0]+r[2])/2, (r[1]+r[3])/2
    
    ent = engine.normalize_path(d, i, (0.9, 1.1))
    if ent is None:
        # Check if it sits in the relative "slot zone" loosely
        rel_x = cx / page.rect.width
        rel_y = cy / page.rect.height
        
        if 0.25 < rel_x < 0.55 and 0.40 < rel_y < 0.60:
            print(f"DROPPED AT PARSE (WHITELIST/RECT): Centroid ({cx:.1f}, {cy:.1f}), Rect Width: {r.width:.1f}, Height: {r.height:.1f}, RelX {rel_x:.2f}, RelY {rel_y:.2f}")

doc1.close()
