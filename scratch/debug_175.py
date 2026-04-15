import sys; sys.path.insert(0, '.')
import fitz, cv2, numpy as np
from raster_diff import render_page_gray, compute_live_zone_px, crop_to_live_zone
from stage2_vector import Stage2Engine

doc1 = fitz.open('Drawings/PRV73B124139.PDF')
doc2 = fitz.open('Drawings/PRV73B124139 - Copy.PDF')
s2 = Stage2Engine()
bounds_v1 = s2.detect_boundaries(doc1[0])
lz = compute_live_zone_px(doc1[0], bounds_v1, 300)
scale = 300/72.0

# Where is 175.65 dimension line in V2?
page2 = doc2[0]
t = page2.get_text('dict')
for b in t.get('blocks',[]):
    if 'lines' not in b: continue
    for l in b['lines']:
        for s in l['spans']:
            if '175' in s['text']:
                bb = s['bbox']
                px_x = bb[0]*scale
                px_y = bb[1]*scale
                print(f'Found "{s["text"]}" at pdf_bbox={[round(x,1) for x in bb]}')
                print(f'  px position: ({px_x:.0f},{px_y:.0f})')
                in_lz = lz[0]<=px_x<=lz[2] and lz[1]<=px_y<=lz[3]
                print(f'  Live zone px: {lz}')
                print(f'  In live zone: {in_lz}')
                crop_x = px_x - lz[0]
                crop_y = px_y - lz[1]
                print(f'  Crop-relative pos: ({crop_x:.0f},{crop_y:.0f})')
                print()

# Also check the live zone in PDF points
print(f'Live zone PDF pts: ({lz[0]/scale:.1f}, {lz[1]/scale:.1f}, {lz[2]/scale:.1f}, {lz[3]/scale:.1f})')
print(f'Page size PDF pts: ({doc2[0].rect.width:.1f}, {doc2[0].rect.height:.1f})')
