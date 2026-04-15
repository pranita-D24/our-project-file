# preprocessor.py — v3.0
# Adaptive preprocessing that works on ANY diagram type:
# - Engineering drawings (CAD, blue/white, black/white)
# - Architectural drawings
# - Schematics and circuit diagrams
# - Scanned documents (yellowed, low contrast)
# - Photos of drawings (variable lighting)

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import logging
from config import (
    TARGET_SIZE, CONTRAST_FACTOR, IMAGE_DIR
)

logger = logging.getLogger(__name__)


class ImagePreprocessor:

    def __init__(self):
        logger.info("ImagePreprocessor v3.0 initialized")

    # ──────────────────────────────────────
    # LOAD
    # ──────────────────────────────────────
    def load_image(self, image_path):
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image not found: {image_path}")
                return None
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Cannot read: {image_path}")
                return None
            logger.info(f"Loaded: {os.path.basename(image_path)} {img.shape}")
            return img
        except Exception as e:
            logger.error(f"Load error: {e}")
            return None

    # ──────────────────────────────────────
    # RESIZE
    # ──────────────────────────────────────
    def resize(self, image, size=TARGET_SIZE):
        try:
            if image is None:
                return None
            return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            logger.error(f"Resize error: {e}")
            return None

    # ──────────────────────────────────────
    # GRAYSCALE
    # ──────────────────────────────────────
    def to_grayscale(self, image):
        try:
            if image is None:
                return None
            if len(image.shape) == 2:
                return image
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"Grayscale error: {e}")
            return None

    # ──────────────────────────────────────
    # DETECT DIAGRAM TYPE
    # ──────────────────────────────────────
    def detect_diagram_type(self, gray):
        """
        Automatically detects what kind of diagram this is
        so preprocessing parameters can be tuned accordingly.

        Returns: "cad_white", "cad_dark", "scanned",
                 "photo", "blueprint", "unknown"
        """
        if gray is None:
            return "unknown"

        mean_val  = float(np.mean(gray))
        std_val   = float(np.std(gray))
        dark_pct  = float(np.sum(gray < 50)  / gray.size * 100)
        light_pct = float(np.sum(gray > 200) / gray.size * 100)

        # Blueprint: dark background, light lines
        if mean_val < 80 and light_pct < 30:
            return "blueprint"

        # CAD white: mostly white background, thin dark lines
        if light_pct > 70 and dark_pct < 10:
            return "cad_white"

        # Scanned: medium gray background, moderate contrast
        if 100 < mean_val < 180 and std_val < 60:
            return "scanned"

        # Photo: high variance, mixed tones
        if std_val > 70:
            return "photo"

        # CAD dark or unknown
        if mean_val < 120:
            return "cad_dark"

        return "cad_white"

    # ──────────────────────────────────────
    # ADAPTIVE CONTRAST ENHANCEMENT
    # ──────────────────────────────────────
    def enhance_contrast_adaptive(self, gray):
        """
        Adapts contrast enhancement to the diagram type.
        Uses CLAHE for scanned/photo, PIL enhance for CAD.
        This is the key fix for burn-out on light drawings
        and under-enhancement on dark ones.
        """
        if gray is None:
            return None

        dtype = self.detect_diagram_type(gray)
        logger.info(f"Diagram type detected: {dtype}")

        if dtype == "blueprint":
            # Invert so lines become dark, then CLAHE
            inv   = cv2.bitwise_not(gray)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(inv)

        elif dtype == "scanned":
            # CLAHE works best for uneven scans
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
            return clahe.apply(gray)

        elif dtype == "photo":
            # Strong CLAHE + normalize
            clahe   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        elif dtype == "cad_white":
            # Mild enhancement — don't burn out
            pil_img  = Image.fromarray(gray)
            enhancer = ImageEnhance.Contrast(pil_img)
            enhanced = enhancer.enhance(1.3)   # was hardcoded 1.5
            return np.array(enhanced)

        else:
            # Default CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            return clahe.apply(gray)

    # ──────────────────────────────────────
    # NOISE REMOVAL
    # ──────────────────────────────────────
    def remove_noise(self, image):
        """Light bilateral filter — preserves thin CAD lines."""
        try:
            if image is None:
                return None
            if len(image.shape) == 2:
                return cv2.bilateralFilter(
                    image, d=5, sigmaColor=35, sigmaSpace=35)
            b, g, r = cv2.split(image)
            b = cv2.bilateralFilter(b, 5, 35, 35)
            g = cv2.bilateralFilter(g, 5, 35, 35)
            r = cv2.bilateralFilter(r, 5, 35, 35)
            return cv2.merge([b, g, r])
        except Exception as e:
            logger.error(f"Noise removal error: {e}")
            return None

    # ──────────────────────────────────────
    # SHARPEN LINES
    # ──────────────────────────────────────
    def sharpen_lines(self, image):
        """Unsharp mask to restore thin stroke clarity after resize."""
        try:
            if image is None:
                return None
            gray  = self.to_grayscale(image) \
                if len(image.shape) == 3 else image
            blur  = cv2.GaussianBlur(gray, (0,0), sigmaX=1.2)
            sharp = cv2.addWeighted(gray, 1.4, blur, -0.4, 0)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            if len(image.shape) == 3:
                return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
            return sharp
        except Exception as e:
            logger.error(f"Sharpen error: {e}")
            return image

    # ──────────────────────────────────────
    # NORMALIZE
    # ──────────────────────────────────────
    def normalize(self, image):
        try:
            if image is None:
                return None
            return cv2.normalize(image, None,
                                 alpha=0, beta=255,
                                 norm_type=cv2.NORM_MINMAX)
        except Exception as e:
            logger.error(f"Normalize error: {e}")
            return None

    # ──────────────────────────────────────
    # DESKEW
    # ──────────────────────────────────────
    def deskew(self, image):
        """Fix slightly tilted scanned drawings (up to ~5°)."""
        try:
            if image is None:
                return None
            gray  = self.to_grayscale(image)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
            if lines is None:
                return image
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = (theta * 180 / np.pi) - 90
                if abs(angle) < 45:
                    angles.append(angle)
            if not angles:
                return image
            median_angle = np.median(angles)
            if abs(median_angle) < 0.3:
                return image
            h, w   = image.shape[:2]
            center = (w//2, h//2)
            M      = cv2.getRotationMatrix2D(
                center, median_angle, 1.0)
            return cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.error(f"Deskew error: {e}")
            return image

    # ──────────────────────────────────────
    # AUTO BORDER CROP
    # ──────────────────────────────────────
    def auto_crop_border(self, gray, margin=0.01):
        """
        Detects and removes scan borders / blank margins.
        Returns the crop bounding box (not the cropped image)
        so callers can offset coordinates correctly.
        """
        try:
            _, binary = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            coords = cv2.findNonZero(binary)
            if coords is None:
                return 0, 0, gray.shape[1], gray.shape[0]
            x, y, w, h = cv2.boundingRect(coords)
            H, W = gray.shape
            mx   = int(W * margin)
            my   = int(H * margin)
            x1   = max(0, x - mx)
            y1   = max(0, y - my)
            x2   = min(W, x + w + mx)
            y2   = min(H, y + h + my)
            return x1, y1, x2 - x1, y2 - y1
        except Exception as e:
            logger.error(f"Border crop error: {e}")
            return 0, 0, gray.shape[1], gray.shape[0]

    # ──────────────────────────────────────
    # FULL SINGLE-IMAGE PIPELINE
    # ──────────────────────────────────────
    def preprocess(self, image_path,
                   do_deskew=True,
                   do_denoise=True,
                   do_normalize=True):
        try:
            img = self.load_image(image_path)
            if img is None:
                return None
            img = self.resize(img)
            if do_deskew:
                img = self.deskew(img)
            if do_denoise:
                img = self.remove_noise(img)
            img = self.to_grayscale(img)
            if img is None:
                return None
            img = self.enhance_contrast_adaptive(img)
            img = self.sharpen_lines(img)
            if do_normalize:
                img = self.normalize(img)
            logger.info(f"Preprocessed: {os.path.basename(image_path)}")
            return img
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return None

    # ──────────────────────────────────────
    # PAIR PIPELINE
    # ──────────────────────────────────────
    def preprocess_pair(self, path1, path2):
        """
        Preprocess two images as a matched pair.
        Both get identical pipeline steps so they
        are comparable. V1=path1, V2=path2 always.
        """
        try:
            logger.info("Preprocessing image pair...")

            img1 = self.load_image(path1)
            img2 = self.load_image(path2)
            if img1 is None or img2 is None:
                logger.error("Failed to load one or both images")
                return None, None

            # Resize to same standard size
            img1 = self.resize(img1)
            img2 = self.resize(img2)

            # Deskew
            img1 = self.deskew(img1)
            img2 = self.deskew(img2)

            # Denoise (light)
            img1 = self.remove_noise(img1)
            img2 = self.remove_noise(img2)

            # Grayscale
            img1 = self.to_grayscale(img1)
            img2 = self.to_grayscale(img2)

            # Detect diagram type from V1 and apply SAME
            # settings to both (critical for fair comparison)
            dtype = self.detect_diagram_type(img1)

            if dtype == "blueprint":
                img1 = cv2.bitwise_not(img1)
                img2 = cv2.bitwise_not(img2)
                clahe = cv2.createCLAHE(2.0, (8,8))
                img1  = clahe.apply(img1)
                img2  = clahe.apply(img2)

            elif dtype == "scanned":
                clahe = cv2.createCLAHE(3.0, (16,16))
                img1  = clahe.apply(img1)
                img2  = clahe.apply(img2)

            elif dtype == "photo":
                clahe = cv2.createCLAHE(4.0, (8,8))
                img1  = clahe.apply(img1)
                img2  = clahe.apply(img2)
                img1  = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX)
                img2  = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

            else:  # cad_white, cad_dark, unknown
                # Mild PIL contrast
                for ref, img in [(img1, 1), (img2, 2)]:
                    pil = Image.fromarray(ref)
                    enh = ImageEnhance.Contrast(pil).enhance(1.3)
                    if img == 1:
                        img1 = np.array(enh)
                    else:
                        img2 = np.array(enh)

            # Sharpen lines
            img1 = self.sharpen_lines(img1)
            img2 = self.sharpen_lines(img2)

            # Normalize
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)

            logger.info(f"Pair preprocessing complete ✅ (type={dtype})")
            return img1, img2

        except Exception as e:
            logger.error(f"Pair preprocessing failed: {e}")
            import traceback; traceback.print_exc()
            return None, None