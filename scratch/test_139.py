"""Quick test: Run Stage4 fix on PRV73B124139 and print results."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from comparator import compare, simple_report

v1 = r"C:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139.PDF"
v2 = r"C:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139 - Copy.PDF"

print("=" * 60)
print("TESTING PRV73B124139 with Stage4 matching fixes")
print("=" * 60)

result = compare(v1, v2, drawing_id="PRV73B124139")
simple_report(result)

g = result.geometry if result.geometry else {}

print(f"\nVerdict: {result.verdict}")
print(f"Processing Time: {result.processing_time:.2f}s")

# Detail on ADDED items
added = g.get("added", [])
print(f"\n--- ADDED Details ({len(added)} items) ---")
for i, a in enumerate(added):
    c = a.get("centroid", [0,0])
    bbox = a.get("bbox", [0,0,0,0])
    t = a.get("type", "?")
    status = a.get("status", "?")
    region = a.get("region", "?")
    bbox_w = bbox[2]-bbox[0] if len(bbox) >= 4 else 0
    bbox_h = bbox[3]-bbox[1] if len(bbox) >= 4 else 0
    print(f"  [{i}] type={t} status={status} region={region}")
    print(f"       centroid=({c[0]:.1f}, {c[1]:.1f})  bbox_size=({bbox_w:.1f}x{bbox_h:.1f})")

# Detail on REMOVED items
removed = g.get("removed", [])
print(f"\n--- REMOVED Details ({len(removed)} items) ---")
for i, r in enumerate(removed[:10]):  # cap at 10
    c = r.get("centroid", [0,0])
    t = r.get("type", "?")
    region = r.get("region", "?")
    print(f"  [{i}] type={t} region={region} centroid=({c[0]:.1f}, {c[1]:.1f})")

# Resized
resized = g.get("resized", [])
print(f"\n--- RESIZED Details ({len(resized)} items) ---")
for i, rs in enumerate(resized[:5]):
    print(f"  [{i}] type={rs.get('type','?')} centroid={rs.get('centroid')}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
