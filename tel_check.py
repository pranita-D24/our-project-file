import cv2, numpy as np

img = cv2.imread(r'C:\Trivim Internship\drawing files\PRQ93B101905_17_V1.png', 0)
print(f'Image shape: {img.shape}')

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
print(f'Total contours: {len(areas)}')
print(f'Top 10 areas: {areas[:10]}')
print(f'Min area seen: {min(areas)}')
print(f'Max area seen: {max(areas)}')
