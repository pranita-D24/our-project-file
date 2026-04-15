import cv2
import fitz
import numpy as np
import os

DPI = 150

def analyze_diag40():
    v1_path = r"Drawings\PRV73B124140 - Copy.PDF"
    v2_path = r"Drawings\PRV73B124140.PDF"
    
    doc1 = fitz.open(v1_path)
    doc2 = fitz.open(v2_path)
    mat = fitz.Matrix(DPI/72, DPI/72)
    
    pix1 = doc1[0].get_pixmap(matrix=mat)
    pix2 = doc2[0].get_pixmap(matrix=mat)
    
    img1 = np.frombuffer(pix1.samples, dtype=np.uint8).reshape(pix1.h, pix1.w, pix1.n)
    img2 = np.frombuffer(pix2.samples, dtype=np.uint8).reshape(pix2.h, pix2.w, pix2.n)
    
    # Ensure same size for diff
    h_min = min(img1.shape[0], img2.shape[0])
    w_min = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h_min, :w_min, :3]
    img2 = img2[:h_min, :w_min, :3]
    
    # Convert to grayscale for diff
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # NEW PIXELS: V2 is darker than V1 (ink added)
    # Using simple subtraction or absdiff
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    page_h, page_w = thresh.shape
    
    # QUADRANT 1: Top Right (x > 50%, y < 40%)
    roi_tr = thresh[0:int(page_h*0.4), int(page_w*0.5):]
    cnts_tr, _ = cv2.findContours(roi_tr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tr_boxes = []
    for c in cnts_tr:
        if cv2.contourArea(c) > 10:
            x, y, w, h = cv2.boundingRect(c)
            tr_boxes.append([x + int(page_w*0.5), y, w, h])
            
    # STRIP: Top (y < 8%)
    roi_ts = thresh[0:int(page_h*0.08), :]
    cnts_ts, _ = cv2.findContours(roi_ts, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ts_boxes = []
    for c in cnts_ts:
        if cv2.contourArea(c) > 10:
            x, y, w, h = cv2.boundingRect(c)
            ts_boxes.append([x, y, w, h])
            
    print(f"Top-Right Additions (3D area): {len(tr_boxes)}")
    if tr_boxes:
        tr_boxes = np.array(tr_boxes)
        x_min = np.min(tr_boxes[:, 0])
        y_min = np.min(tr_boxes[:, 1])
        x_max = np.max(tr_boxes[:, 0] + tr_boxes[:, 2])
        y_max = np.max(tr_boxes[:, 1] + tr_boxes[:, 3])
        print(f"  3D Bounding Box: x={x_min} y={y_min} w={x_max-x_min} h={y_max-y_min}")
        
    print(f"Top-Strip Additions (NOTE area): {len(ts_boxes)}")
    if ts_boxes:
        ts_boxes = np.array(ts_boxes)
        x_min = np.min(ts_boxes[:, 0])
        y_min = np.min(ts_boxes[:, 1])
        x_max = np.max(ts_boxes[:, 0] + ts_boxes[:, 2])
        y_max = np.max(ts_boxes[:, 1] + ts_boxes[:, 3])
        print(f"  NOTE Bounding Box: x={x_min} y={y_min} w={x_max-x_min} h={y_max-y_min}")

    doc1.close(); doc2.close()

if __name__ == '__main__':
    analyze_diag40()
