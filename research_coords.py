import fitz
import sys
import os

# Add project root to path
sys.path.insert(0, r'C:\Trivim Internship\engineering_comparison_system')

from stage2_vector import Stage2Engine

def research_140():
    p1_path = r'Drawings\PRV73B124140.PDF'
    doc1 = fitz.open(p1_path)
    page1 = doc1[0]
    
    print(f"Page Rect: {page1.rect}")
    
    text_dict = page1.get_text("dict")
    found = False
    for b in text_dict.get("blocks", []):
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                if "FASTENER" in s["text"]:
                    print(f"Found Note: '{s['text']}' at BBox: {s['bbox']}")
                    found = True
    
    if not found:
        print("Note 'FASTENER' not found in text.")

    s2 = Stage2Engine()
    bounds = s2.detect_boundaries(page1)
    print(f"Detected Boundaries: {bounds}")
    
    # Check if a point at x=26 is excluded
    f = bounds["outer_frame"]
    test_x, test_y = 26, 400
    is_excluded = not (f[0]+5 <= test_x <= f[2]-5 and f[1]+5 <= test_y <= f[3]-5)
    print(f"Point (26, 400) excluded by Page Margin Filter? {is_excluded}")

if __name__ == "__main__":
    research_140()
