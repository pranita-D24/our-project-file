import fitz
import os

files = [
    'Drawings/PRQ93B101928 (1).pdf',
    'Drawings/PRV73B124140.PDF',
    'Drawings/PRV73B124138.PDF'
]

for f in files:
    if not os.path.exists(f): 
        print(f"File {f} not found.")
        continue
    doc = fitz.open(f)
    page = doc[0]
    text = page.get_text("text").strip()
    drawings = page.get_drawings()
    images = page.get_images()
    
    print(f"--- {f} ---")
    print(f"Text length: {len(text)}")
    print(f"Drawings: {len(drawings)}")
    print(f"Images: {len(images)}")
    if len(text) == 0 and len(drawings) == 0 and len(images) > 0:
        print("RESULT: RASTER")
    else:
        print("RESULT: VECTOR")
    print("")
