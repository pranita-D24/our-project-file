import cv2
import fitz
import numpy as np
import os

DPI = 150

def analyze_diag40_refined():
    v1_path = r"Drawings\PRV73B124140 - Copy.PDF"
    v2_path = r"Drawings\PRV73B124140.PDF"
    
    doc1 = fitz.open(v1_path)
    doc2 = fitz.open(v2_path)
    mat = fitz.Matrix(DPI/72, DPI/72)
    
    pix1 = doc1[0].get_pixmap(matrix=mat)
    pix2 = doc2[0].get_pixmap(matrix=mat)
    
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.h, pix1.w, 3)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, 3)
    
    # Grayscale diff
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    page_h, page_w = thresh.shape
    
    # 1. TOP STRIP (y < 8%) - Split Note and 3D View
    roi_top = thresh[0:int(page_h*0.08), :]
    # Vertical projection for splitting
    v_proj = np.sum(roi_top, axis=0)
    
    # Find gaps > 50px (approx 24pt)
    segments = []
    start = -1
    gap_count = 0
    GAP_LIMIT = 50
    for i in range(len(v_proj)):
        if v_proj[i] > 0:
            if start == -1: start = i
            gap_count = 0
        else:
            if start != -1:
                gap_count += 1
                if gap_count > GAP_LIMIT:
                    segments.append((start, i - gap_count))
                    start = -1
    if start != -1: segments.append((start, len(v_proj)-1))
    
    top_boxes = []
    for s1, s2 in segments:
        roi_sub = roi_top[:, s1:s2]
        cnts, _ = cv2.findContours(roi_sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Get tight sub-box
            all_pts = np.concatenate(cnts)
            sx, sy, sw, sh = cv2.boundingRect(all_pts)
            top_boxes.append({"x": sx + s1, "y": sy, "w": sw, "h": sh})
            
    # 2. BOTTOM ZONE (y > 75%)
    roi_bottom = thresh[int(page_h*0.75):, :]
    cnts_bottom, _ = cv2.findContours(roi_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottom_boxes = []
    for c in cnts_bottom:
        if cv2.contourArea(c) > 500:
            bx, by, bw, bh = cv2.boundingRect(c)
            bottom_boxes.append({"x": bx, "y": by + int(page_h*0.75), "w": bw, "h": bh})
            
    # Classify Top Boxes
    note_box = None
    iso_box = None
    for b in top_boxes:
        # Left part (Note): x < 0.7 * page_w
        if b["x"] < page_w * 0.7 and (note_box is None or b["x"] < note_box["x"]):
            note_box = b
        # Right part (Iso): x > 0.6 * page_w
        if b["x"] > page_w * 0.6:
            iso_box = b
            
    # Output Coordinates
    print("--- Final Coordinates Diagram 40 ---")
    if note_box:
        print(f"1. NOTE Text (Top Left): x={note_box['x']} y={note_box['y']} w={note_box['w']} h={note_box['h']}")
    if iso_box:
        print(f"2. 3D Isometric View (Top Right): x={iso_box['x']} y={iso_box['y']} w={iso_box['w']} h={iso_box['h']}")
    for i, b in enumerate(bottom_boxes):
        print(f"3. Bottom Change (Zone {i+1}): x={b['x']} y={b['y']} w={b['w']} h={b['h']}")
        
    doc1.close(); doc2.close()

if __name__ == '__main__':
    analyze_diag40_refined()
