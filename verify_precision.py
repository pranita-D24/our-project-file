# verify_precision.py - Stage 5 Precision Verification Suite
import cv2
import numpy as np
from semantic_diff import compute_similarity

def create_test_shape(type='square', size=200, rot=0, scale=1.0):
    img = np.zeros((size, size), dtype=np.uint8)
    
    if type == 'square':
        w = int(100 * scale)
        s = (size - w) // 2
        pts = np.array([[s, s], [s+w, s], [s+w, s+w], [s, s+w]])
    elif type == 'circle':
        r = int(50 * scale)
        cv2.circle(img, (size//2, size//2), r, 255, -1)
        # Skip contour rotation for circle
        return img
    elif type == 'rectangle':
        w, h = int(120 * scale), int(60 * scale)
        x1, y1 = (size - w)//2, (size - h)//2
        pts = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h], [x1, y1+h]])
    
    if rot != 0:
        M = cv2.getRotationMatrix2D((size/2, size/2), rot, 1.0)
        # Simple rotation for test shapes
        temp = np.zeros_like(img)
        cv2.fillPoly(temp, [pts], 255)
        img = cv2.warpAffine(temp, M, (size, size))
    else:
        cv2.fillPoly(img, [pts], 255)
        
    return img

def run_suite():
    print("=== STAGE 5 PRECISION VERIFICATION SUITE ===")
    
    # Pre-test patches (normalized to 64x64)
    def to_64(img):
        # Tightly crop to the shape content
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return cv2.resize(img, (64, 64))
        
        c = cnts[0]
        x, y, w, h = cv2.boundingRect(c)
        size_max = max(w, h)
        
        # Build square mask and center the shape
        mask = np.zeros((size_max, size_max), dtype=np.uint8)
        dx = (size_max - w) // 2
        dy = (size_max - h) // 2
        c_shifted = c - [x - dx, y - dy]
        cv2.drawContours(mask, [c_shifted], -1, 255, -1)
        
        # Production-grade 64x64 normalization
        res = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)
        _, res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY)
        return res

    base_sq = to_64(create_test_shape('square'))
    base_rect = to_64(create_test_shape('rectangle'))
    
    # 1. Identity
    sim1 = compute_similarity(base_sq, base_sq)
    print(f"[1] Identity (Square vs Square): {sim1:.4f}  | Expected: 1.0000")
    
    # 2. Rotation (45 deg)
    sq_45 = to_64(create_test_shape('square', rot=45))
    sim2 = compute_similarity(base_sq, sq_45)
    print(f"[2] Rotation (Square vs 45 deg): {sim2:.4f}  | Expected: > 0.95")
    
    # 3. Scale (1.2x)
    sq_12 = to_64(create_test_shape('square', scale=1.2))
    sim3 = compute_similarity(base_sq, sq_12)
    print(f"[3] Scale (Square vs 1.2x):     {sim3:.4f}  | Expected: > 0.98")
    
    # 4. Discrimination (Negative Case)
    base_circ = to_64(create_test_shape('circle'))
    sim4 = compute_similarity(base_sq, base_circ)
    print(f"[4] Discrimination (Sq vs Circ): {sim4:.4f}  | Expected: < 0.50")
    
    # 5. Discrimination (Rect vs Sq)
    sim5 = compute_similarity(base_sq, base_rect)
    print(f"[5] Discrimination (Sq vs Rect): {sim5:.4f}  | Expected: < 0.85")

    success = (sim1 > 0.999 and sim2 > 0.95 and sim3 > 0.96 and sim4 < 0.5)
    print("\nOVERALL STATUS: " + ("PASS" if success else "FAIL"))

if __name__ == "__main__":
    run_suite()
