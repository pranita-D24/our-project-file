"""Check if 175.65 dimension shows up in dimension comparison."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from comparator import compare

v1 = r"C:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139.PDF"
v2 = r"C:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139 - Copy.PDF"

result = compare(v1, v2, drawing_id="PRV73B124139")

# Check dimensions
dims = result.processing_info.get("dimensions", {})
print("=== DIMENSION COMPARISON ===")
print(f"Added dims:    {dims.get('added', [])}")
print(f"Removed dims:  {dims.get('removed', [])}")
print(f"Modified dims: {dims.get('modified', [])}")
print(f"Dim changes (d_mod): {result.dim_changes}")

# Check removed text-notes
g = result.geometry or {}
removed = g.get("removed", [])
print(f"\n=== REMOVED items (type=text-note) ===")
for r in removed:
    if r.get("type") == "text-note":
        print(f"  {r.get('notes', '')}")
