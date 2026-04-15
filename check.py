
import cv2
import numpy
import sklearn
import easyocr
import streamlit
import scipy
import shapely
import matplotlib
import reportlab
import imutils

print("OpenCV version:", cv2.__version__)
print("SIFT available:", hasattr(cv2, 'SIFT_create'))
print("EasyOCR version:", easyocr.__version__)
print("All imports OK ✓")
