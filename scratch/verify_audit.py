import sys
sys.path.insert(0, '.')
import comparator
import stage5_moves
import fitz

def verify_138():
    p1 = 'Drawings/PRV73B124138.PDF'
    p2 = 'Drawings/PRV73B124138 - Copy.PDF'
    engine = comparator.Stage2Engine()
    
    doc1 = fitz.open(p1)
    doc2 = fitz.open(p2)
    
    f1 = engine.detect_boundaries(doc1[0])["outer_frame"]
    f2 = engine.detect_boundaries(doc2[0])["outer_frame"]
    
    print(f"PRV73B124138 Frames:")
    print(f"  V1 (Original) Frame: {f1}")
    print(f"  V2 (Copy) Frame:     {f2}")
    
    # Check 60 position
    t1 = doc1[0].get_text("dict")
    for block in t1["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if "60" in span["text"]:
                    b = span["bbox"]
                    in_f1 = (b[0] >= f1[0]-2 and b[2] <= f1[2]+2 and b[1] >= f1[1]-2 and b[3] <= f1[3]+2)
                    print(f"  60 in Original: BBox={b}, In Frame={in_f1}")

def verify_140():
    p1 = 'Drawings/PRV73B124140.PDF'
    p2 = 'Drawings/PRV73B124140 - Copy.PDF'
    res = comparator.compare(p1, p2, drawing_id='TEST_140')
    added = res.geometry.get('added', [])
    
    print(f"\nPRV73B124140 Added Clusters (Pre-Refinement):")
    for c in added:
        print(f"  Region: {c.get('region')}, Status: {c.get('status')}, Prims: {c.get('primitive_count')}")

if __name__ == "__main__":
    verify_138()
    verify_140()
