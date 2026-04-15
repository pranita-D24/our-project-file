import sys, fitz
sys.path.append('c:/Trivim Internship/engineering_comparison_system')
import raster_diff, stage2_vector

v1_path = r'c:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139.PDF'
v2_path = r'c:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139 - Copy.PDF'

doc1 = fitz.open(v1_path)
doc2 = fitz.open(v2_path)
page1, page2 = doc1[0], doc2[0]

s2 = stage2_vector.Stage2Engine()
bounds_v1 = s2.detect_boundaries(page1)

live_zone_y1 = bounds_v1['live_zone'][3]

print('1. Validation Checks:')
print(f'   - live_zone.y1: {live_zone_y1:.2f}pt')

res = raster_diff.raster_compare(page1, page2, bounds_v1, bounds_v1, dpi=300, intensity_gap=16, min_area_pct=0.00005)
added = res['geometry']['added']

dim_175_y = 707.04
print(f'   - dim_175 blob y-coordinate ({dim_175_y}pt) is INSIDE live_zone: {dim_175_y <= live_zone_y1}')

print('\n2. Blob detection (Pass 1):')
print(f'   - blobs_found after area filter: {res["debug"]["after_area_filter"]}')
print(f'   - added_boxes: {len(added)}')

for i, a in enumerate(added):
    print(f'   - Box {i+1}: {a["bbox"]}')
