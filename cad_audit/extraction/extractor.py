import fitz
from .normalizer import normalize_cad_text

RENDER_DPI = 300
ZOOM = RENDER_DPI / 72  # 4.1667 — use everywhere

def extract_drawing(pdf_path: str) -> list:
    """Extracts high-fidelity image and coordinate-locked spans from PDF."""
    doc = fitz.open(pdf_path)
    results = []
    
    for page in doc:
        # Render image at 300 DPI
        mat = fitz.Matrix(ZOOM, ZOOM)
        pix = page.get_pixmap(matrix=mat)
        
        # Extract all text spans WITH bbox
        spans = []
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = normalize_cad_text(span["text"])
                    if not text:
                        continue
                    # CRITICAL: store zoomed bbox immediately
                    b = span["bbox"]
                    spans.append({
                        "text": text,
                        "bbox": (b[0]*ZOOM, b[1]*ZOOM, 
                                 b[2]*ZOOM, b[3]*ZOOM),
                        "centroid": (
                            ((b[0]+b[2])/2) * ZOOM,
                            ((b[1]+b[3])/2) * ZOOM
                        ),
                        "page_height": page.rect.height * ZOOM
                    })
        results.append({
            "page": page.number,
            "image": pix,
            "spans": spans
        })
    return results
