# segmentor.py - Stage 4 Hierarchical Component Segmentor
import cv2
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class ComponentSegmentor:
    def __init__(self, min_area=150, max_area=900000):
        self.min_area = min_area
        self.max_area = max_area

    def extract_components(self, binary_image):
        """
        Extracts components using topological hierarchy (RETR_CCOMP).
        Returns list of component dicts with normalized 64x64 patches.
        """
        # Find contours with hierarchy
        # RETR_CCOMP: organizes contours into a 2-level hierarchy (external and holes)
        _t = time.time()
        cnts, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print(f"[TEL] contours={len(cnts)} t={time.time()-_t:.3f}s")
        
        components = []
        if hierarchy is None:
            return components

        hierarchy = hierarchy[0] # Squeeze first dimension
        
        for i, (cnt, h) in enumerate(zip(cnts, hierarchy)):
            # Only process top-level contours (extreme outer boundaries)
            # hierarchy[i][3] == -1 means it has no parent
            if h[3] != -1:
                continue
                
            area = cv2.contourArea(cnt)
            if not (self.min_area <= area <= self.max_area):
                continue
                
            x, y, w, h_rect = cv2.boundingRect(cnt)
            centroid = (x + w/2, y + h_rect/2)
            
            # Create a localized mask for this component including its holes
            # We want only this specific hierarchy branch (the parent and its children)
            # Create a square mask to preserve aspect ratio (User requirement for precision)
            size_max = max(w, h_rect)
            mask = np.zeros((size_max, size_max), dtype=np.uint8)
            
            # Center the contour in the square mask
            dx = (size_max - w) // 2
            dy = (size_max - h_rect) // 2
            cnt_shifted = cnt - [x - dx, y - dy]
            cv2.drawContours(mask, [cnt_shifted], -1, 255, -1)
            
            # Find and cut out holes (children of this contour)
            child_idx = h[2]
            while child_idx != -1:
                child_cnt = cnts[child_idx] - [x - dx, y - dy]
                cv2.drawContours(mask, [child_cnt], -1, 0, -1)
                child_idx = hierarchy[child_idx][0]

            # Normalization to 64x64 for Layer 5 - Now aspect-ratio aware
            patch = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)
            _, patch = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY)
            
            components.append({
                "id": i,
                "bbox": (x, y, w, h_rect),
                "centroid": centroid,
                "area": area,
                "aspect_ratio": w / (h_rect + 1e-6),
                "patch_64": patch,
                "contour": cnt
            })
            
        return components

def extract_mechanical_components(binary_img, profile=None):
    """Bridge for the comparator integration."""
    min_a = profile.min_component_area if profile else 800
    max_a = profile.max_component_area if profile else 900000
    seg = ComponentSegmentor(min_area=min_a, max_area=max_a)
    return seg.extract_components(binary_img)