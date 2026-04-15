import os
import shutil
import json
from pathlib import Path
import fitz
import numpy as np
from stage2_vector import Stage2Engine

# ═══════════════════════════════════════════════════════════
# TEST SETUP
# ═══════════════════════════════════════════════════════════

TEST_ROOT = Path("c:/Trivim Internship/engineering_comparison_system/test_stage2_data")
TEST_OUT = TEST_ROOT / "output"

def create_test_drawing(path: Path, items: list):
    doc = fitz.open()
    page = doc.new_page(width=1000, height=800)
    
    # Draw a Frame (Outer boundary)
    page.draw_rect((50, 50, 950, 750), width=1.5, color=(0,0,0))
    
    for item_type, data in items:
        if item_type == "text":
            page.insert_text(data["pos"], data["text"], fontsize=data.get("size", 10))
        elif item_type == "circle":
            page.draw_circle(data["center"], data["radius"], width=data.get("width", 0.5))
        elif item_type == "line":
            page.draw_line(data["p1"], data["p2"], width=data.get("width", 0.5))
        elif item_type == "rect":
            page.draw_rect(data["rect"], width=data.get("width", 0.5))
            
    doc.save(path)
    doc.close()

def setup_test_env():
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)
    
    # V1: Base drawing
    items_v1 = [
        # 1. Noise: File path outside frame
        ("text", {"pos": (500, 780), "text": "C:/Projects/Drawing_V1.pdf", "size": 8}),
        
        # 2. Grid Labels (A, 1) on borders
        ("text", {"pos": (20, 400), "text": "A", "size": 13.9}),
        ("text", {"pos": (500, 20), "text": "1", "size": 13.9}),
        
        # 3. Composite Dimension (Split spans)
        # 12.2 +0.1
        ("text", {"pos": (200, 200), "text": "12.2", "size": 10}),
        ("text", {"pos": (225, 195), "text": "+0.1", "size": 6}),
        ("text", {"pos": (225, 205), "text": "+0.2", "size": 6}),
        
        # 4. Standard Geometry
        ("rect", {"rect": (300, 300, 350, 350), "width": 1.0}),
        
        # 5. Concentric Structural Hole (Should NOT be a balloon)
        ("circle", {"center": (500, 500), "radius": 15, "width": 1.0}),
        ("circle", {"center": (500, 500), "radius": 20, "width": 1.5}),
        
        # 6. Structural Noise (Fillets/Arcs) - 50 items
    ]
    for i in range(50):
        items_v1.append(("circle", {"center": (10 * i, 10), "radius": 1, "width": 0.1})) # Tiny noise
        
    create_test_drawing(TEST_ROOT / "ADV_V1.pdf", items_v1)
    
    # V2: Changes
    # Add 10 consistent balloons to trigger Isolation Forest clustering
    balloons_v2 = []
    for i in range(10):
        balloons_v2.append(("circle", {"center": (100 + i*40, 600), "radius": 12, "width": 0.5}))
        balloons_v2.append(("text", {"pos": (100 + i*40 - 5, 605), "text": str(i+1), "size": 10}))

    items_v2 = [
        # Same noise
        ("text", {"pos": (500, 780), "text": "C:/Projects/Drawing_V1.pdf", "size": 8}),
        ("text", {"pos": (20, 400), "text": "A", "size": 13.9}),
        ("text", {"pos": (500, 20), "text": "1", "size": 13.9}),
        
        # MODIFIED: 12.2 -> 14.5
        ("text", {"pos": (200, 200), "text": "14.5", "size": 10}),
        ("text", {"pos": (225, 195), "text": "+0.1", "size": 6}),
        ("text", {"pos": (225, 205), "text": "+0.2", "size": 6}),
        
        # Concentric Hole preserved
        ("circle", {"center": (500, 500), "radius": 15, "width": 1.0}),
        ("circle", {"center": (500, 500), "radius": 20, "width": 1.5}),
        
        # ADDED: Geometry
        ("rect", {"rect": (300, 300, 350, 350), "width": 1.0}), # Remained
        ("circle", {"center": (800, 600), "radius": 20, "width": 1.0}) # Added structural circle
    ]
    for i in range(50):
        items_v2.append(("circle", {"center": (10 * i, 10), "radius": 1, "width": 0.1}))
        
    items_v2 += balloons_v2
    create_test_drawing(TEST_ROOT / "ADV_V2.pdf", items_v2)

# ═══════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════

def test_stage2_advanced_flow():
    setup_test_env()
    engine = Stage2Engine()
    
    path_v1 = str(TEST_ROOT / "ADV_V1.pdf")
    path_v2 = str(TEST_ROOT / "ADV_V2.pdf")
    
    report = engine.process_pair(path_v1, path_v2, "ADV-DWG", str(TEST_OUT))
    
    print("\n" + "="*40)
    print("ADVANCED STAGE 2 TEST RESULTS")
    print("="*40)
    
    page = report["pages"][0]
    
    # Verify Composite Grouping
    # "12.2+0.1+0.2" should be one dimension entity
    # Modified from 12.2... to 14.5...
    mods = page["dimensions"]["modified"]
    print(f"Dimension Mods: {len(mods)}")
    for m in mods:
        print(f"  Change: {m['from']} -> {m['to']}")
        
    assert len(mods) == 1
    assert "12.2" in mods[0]["from"]
    assert "14.5" in mods[0]["to"]
    
    # Verify Noise Filtering
    # Grid labels (A, 1) and file path should be GONE (not in report)
    # Total entities should be low
    dim_count = len(page["dimensions"]["added"]) + len(page["dimensions"]["removed"]) + len(page["dimensions"]["modified"])
    print(f"Total Dimensions Found: {dim_count}")
    assert dim_count == 1
    
    # Verify Added Geometry
    added_geom = page["geometry"]["added"]
    print(f"Added Geometry: {len(added_geom)}")
    assert len(added_geom) == 1
    
    print("\nALL ADVANCED STAGE 2 TESTS PASSED!")

if __name__ == "__main__":
    test_stage2_advanced_flow()
