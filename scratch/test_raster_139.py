"""Test raster pixel-diff pipeline on PRV73B124139."""
import os, sys
sys.path.insert(0, os.path.abspath('.'))

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

import fitz
from stage2_vector import Stage2Engine
from raster_diff import raster_compare, raster_visual_report

v1_path = r"Drawings\PRV73B124139.PDF"
v2_path = r"Drawings\PRV73B124139 - Copy.PDF"

print("=" * 60)
print("RASTER PIXEL-DIFF TEST: PRV73B124139")
print("=" * 60)

doc1 = fitz.open(v1_path)
doc2 = fitz.open(v2_path)
page1, page2 = doc1[0], doc2[0]

# Get live zone bounds from Stage 2
s2 = Stage2Engine()
bounds_v1 = s2.detect_boundaries(page1)
bounds_v2 = s2.detect_boundaries(page2)

print(f"Live zone V1: {bounds_v1.get('live_zone')}")
print(f"Live zone V2: {bounds_v2.get('live_zone')}")

# Run raster diff
result = raster_compare(page1, page2, bounds_v1, bounds_v2, dpi=300)

geom = result["geometry"]
print(f"\n--- RESULTS ---")
print(f"ADDED:   {len(geom['added'])}")
print(f"REMOVED: {len(geom['removed'])}")

print(f"\n--- ADDED Details ---")
for i, a in enumerate(geom["added"]):
    bbox = a["bbox"]
    c = a["centroid"]
    print(f"  [{i}] centroid=({c[0]:.1f}, {c[1]:.1f})  "
          f"bbox=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})  "
          f"area={a['area']:.0f}pt²  mean_v1={a['mean_v1']:.1f} mean_v2={a['mean_v2']:.1f}")

print(f"\n--- REMOVED Details ---")
for i, r in enumerate(geom["removed"]):
    bbox = r["bbox"]
    c = r["centroid"]
    print(f"  [{i}] centroid=({c[0]:.1f}, {c[1]:.1f})  "
          f"bbox=({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})  "
          f"area={r['area']:.0f}pt²  mean_v1={r['mean_v1']:.1f} mean_v2={r['mean_v2']:.1f}")

print(f"\nDebug: {result['debug']}")

# Generate visual report
out = raster_visual_report(page1, page2, result, "PRV73B124139", output_dir="visuals")
print(f"\nReport: {out}")

doc1.close()
doc2.close()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
