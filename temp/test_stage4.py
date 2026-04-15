import comparator
import fitz

res = comparator.compare('Drawings/PRV73B124139.PDF', 'Drawings/PRV73B124139 - Copy.PDF')
doc = fitz.open('Drawings/PRV73B124139.PDF')
page = doc[0]
w, h = page.rect.width, page.rect.height

# Instead of relying on the final filtered output, let's look straight into the Stage 4 output.
# comparator.compare returns HFReport. 
# Inside comparator we have: stage4.compare_pages(...)
from stage4_geometry import Stage4Engine
import stage2_vector
s4 = Stage4Engine()
s2 = stage2_vector.Stage2Engine()

d1 = s2.extract_page_data(doc[0], "139")
d2 = s2.extract_page_data(fitz.open('Drawings/PRV73B124139 - Copy.PDF')[0], "139-C")

bounds1 = s2.detect_boundaries(doc[0])
bounds2 = s2.detect_boundaries(fitz.open('Drawings/PRV73B124139 - Copy.PDF')[0])

res = s4.compare_pages(doc[0], fitz.open('Drawings/PRV73B124139 - Copy.PDF')[0], 
                       d1["dimensions"], d2["dimensions"], 
                       d1["balloons"], d2["balloons"],
                       bounds1, bounds2)

added = res["geometry"]["added"]
removed = res["geometry"]["removed"]

for pool, name in [(added, "ADDED"), (removed, "REMOVED")]:
    for item in pool:
        cx, cy = item.get("centroid", (0,0))
        rx, ry = cx/w, cy/h
        if 0.25 < rx < 0.55 and 0.40 < ry < 0.60:
            print(f"SURVIVED STAGE 4 {name}:", cx, cy, item.get("type"), item.get("bbox"))
