import cv2
import fitz
import numpy as np
import os

DPI = 150

def global_diagnostic_diag40():
    v1_path = r"Drawings\PRV73B124140 - Copy.PDF"
    v2_path = r"Drawings\PRV73B124140.PDF"
    
    doc1 = fitz.open(v1_path)
    doc2 = fitz.open(v2_path)
    mat = fitz.Matrix(DPI/72, DPI/72)
    
    pix1 = doc1[0].get_pixmap(matrix=mat)
    pix2 = doc2[0].get_pixmap(matrix=mat)
    
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.h, pix1.w, 3)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, 3)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Aggressive diff to find everything
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    
    # Morphology to merge nearby blobs
    k = np.ones((15, 15), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Total Clusters Found: {len(cnts)}")
    page_h, page_w = thresh.shape
    
    all_regions = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            all_regions.append({"x": x, "y": y, "w": w, "h": h, "area": area})
            
    # Sort by y, then x
    all_regions = sorted(all_regions, key=lambda r: (r['y'], r['x']))
    
    for i, r in enumerate(all_regions):
        rel_x = r['x'] / page_w
        rel_y = r['y'] / page_h
        print(f"Cluster {i+1}: x={r['x']} y={r['y']} w={r['w']} h={r['h']} (Area: {r['area']}) [RelX: {rel_x:.2f}, RelY: {rel_y:.2f}]")

    doc1.close(); doc2.close()

if __name__ == '__main__':
    global_diagnostic_diag40()
