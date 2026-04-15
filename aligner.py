# aligner.py — v3.0
# Multi-method alignment with fallback chain:
# 1. SIFT (best for technical drawings, scale-invariant)
# 2. ORB (fast, good for clean drawings)
# 3. Phase correlation (works even with no keypoints)
# 4. ECC (pixel-level refinement)
# 5. Identity (passthrough if all else fails)

import cv2
import numpy as np
import logging
from config import ORB_FEATURES, GOOD_MATCH_RATIO

logger = logging.getLogger(__name__)


class ImageAligner:

    def __init__(self):
        # ORB — fast, good for clean CAD drawings
        self.orb     = cv2.ORB_create(
            ORB_FEATURES,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31)

        # SIFT — scale-invariant, best for varied diagrams
        # Available in opencv-contrib or recent opencv
        try:
            self.sift = cv2.SIFT_create(
                nfeatures=5000,
                contrastThreshold=0.02,
                edgeThreshold=15)
            self.sift_available = True
        except Exception:
            self.sift = None
            self.sift_available = False
            logger.warning("SIFT not available — using ORB only")

        self.bf_orb  = cv2.BFMatcher(cv2.NORM_HAMMING,  crossCheck=True)
        self.bf_sift = cv2.BFMatcher(cv2.NORM_L2,       crossCheck=False)
        logger.info("ImageAligner v3.0 initialized "
                    f"(SIFT={'yes' if self.sift_available else 'no'})")

    # ─────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────
    def _to_gray(self, img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    def _enhance_for_keypoints(self, gray):
        """Boost contrast and edges so keypoints are stable across versions."""
        clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Mild unsharp mask to sharpen line edges
        blur     = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        sharp    = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _validate_homography(self, H, img_shape):
        """
        Reject degenerate homographies that would warp the image
        into garbage (flip, extreme scale, rotation > 15°).
        """
        if H is None:
            return False
        h, w = img_shape[:2]

        # Check corners don't get mapped too far
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warped  = cv2.perspectiveTransform(corners, H)
        warped  = warped.reshape(-1, 2)

        # Each warped corner must stay within 2× the image bounds
        for pt in warped:
            if (pt[0] < -w or pt[0] > 2*w or
                    pt[1] < -h or pt[1] > 2*h):
                return False

        # Determinant check — negative det = flip, near-zero = degenerate
        det = H[0,0]*H[1,1] - H[0,1]*H[1,0]
        if det < 0.1 or det > 10.0:
            return False

        # Rotation angle check (from homography decomposition)
        angle = abs(np.degrees(np.arctan2(H[1,0], H[0,0])))
        if angle > 20.0:   # more than 20° rotation is suspicious
            return False

        return True

    # ─────────────────────────────────────────
    # METHOD 1: SIFT alignment
    # ─────────────────────────────────────────
    def _align_sift(self, orig_gray, mod_gray, orig_shape):
        if not self.sift_available:
            return None

        try:
            e1 = self._enhance_for_keypoints(orig_gray)
            e2 = self._enhance_for_keypoints(mod_gray)

            kp1, des1 = self.sift.detectAndCompute(e1, None)
            kp2, des2 = self.sift.detectAndCompute(e2, None)

            if (des1 is None or des2 is None or
                    len(kp1) < 10 or len(kp2) < 10):
                return None

            # Lowe's ratio test
            matches = self.bf_sift.knnMatch(des1, des2, k=2)
            good    = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.72 * n.distance:
                        good.append(m)

            logger.info(f"SIFT: {len(kp1)}/{len(kp2)} kps, "
                        f"{len(good)} good matches")

            if len(good) < 8:
                return None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

            H, mask = cv2.findHomography(
                pts2, pts1, cv2.RANSAC, 4.0,
                maxIters=3000, confidence=0.995)

            if not self._validate_homography(H, orig_shape):
                return None

            inliers = int(mask.sum()) if mask is not None else 0
            logger.info(f"SIFT homography inliers: {inliers}")

            if inliers < 6:
                return None

            return H

        except Exception as e:
            logger.warning(f"SIFT align failed: {e}")
            return None

    # ─────────────────────────────────────────
    # METHOD 2: ORB alignment
    # ─────────────────────────────────────────
    def _align_orb(self, orig_gray, mod_gray, orig_shape):
        try:
            e1 = self._enhance_for_keypoints(orig_gray)
            e2 = self._enhance_for_keypoints(mod_gray)

            kp1, des1 = self.orb.detectAndCompute(e1, None)
            kp2, des2 = self.orb.detectAndCompute(e2, None)

            if (des1 is None or des2 is None or
                    len(kp1) < 8 or len(kp2) < 8):
                return None

            matches = self.bf_orb.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            n_good  = max(8, int(len(matches) * GOOD_MATCH_RATIO))
            good    = matches[:n_good]

            logger.info(f"ORB: {len(kp1)}/{len(kp2)} kps, "
                        f"{len(good)} good matches")

            if len(good) < 6:
                return None

            pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

            H, mask = cv2.findHomography(
                pts2, pts1, cv2.RANSAC, 5.0,
                maxIters=2000, confidence=0.99)

            if not self._validate_homography(H, orig_shape):
                return None

            inliers = int(mask.sum()) if mask is not None else 0
            logger.info(f"ORB homography inliers: {inliers}")

            if inliers < 5:
                return None

            return H

        except Exception as e:
            logger.warning(f"ORB align failed: {e}")
            return None

    # ─────────────────────────────────────────
    # METHOD 3: Phase correlation (translation only)
    # Works even when keypoint methods fail — finds
    # the XY shift using FFT.
    # ─────────────────────────────────────────
    def _align_phase_correlation(self, orig_gray, mod_gray):
        try:
            f1  = np.float32(orig_gray)
            f2  = np.float32(mod_gray)
            (dx, dy), response = cv2.phaseCorrelate(f1, f2)

            # Ignore tiny shifts — not worth warping
            shift = np.hypot(dx, dy)
            logger.info(f"Phase correlation shift: "
                        f"dx={dx:.1f} dy={dy:.1f} "
                        f"response={response:.3f}")

            if shift < 2.0 or response < 0.02:
                # Images are already well-aligned
                return np.eye(3, dtype=np.float32)

            # Build translation-only homography
            H = np.eye(3, dtype=np.float32)
            H[0, 2] = dx
            H[1, 2] = dy
            return H

        except Exception as e:
            logger.warning(f"Phase correlation failed: {e}")
            return None

    # ─────────────────────────────────────────
    # METHOD 4: ECC (Enhanced Correlation Coeff)
    # Sub-pixel accurate, good for nearly-aligned images.
    # Used as a refinement step after coarse alignment.
    # ─────────────────────────────────────────
    def _refine_ecc(self, orig_gray, mod_gray_warped):
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                50, 1e-4)

            # Downsample for speed
            scale = 0.5
            s1 = cv2.resize(orig_gray,
                            (int(orig_gray.shape[1]*scale),
                             int(orig_gray.shape[0]*scale)))
            s2 = cv2.resize(mod_gray_warped,
                            (int(mod_gray_warped.shape[1]*scale),
                             int(mod_gray_warped.shape[0]*scale)))

            _, warp_matrix = cv2.findTransformECC(
                s1, s2, warp_matrix,
                cv2.MOTION_EUCLIDEAN, criteria,
                inputMask=None, gaussFiltSize=5)

            # Scale translation back
            warp_matrix[0, 2] /= scale
            warp_matrix[1, 2] /= scale

            # Only apply if correction is small (refinement, not redo)
            tx = abs(warp_matrix[0, 2])
            ty = abs(warp_matrix[1, 2])
            if tx > 30 or ty > 30:
                return None  # ECC diverged

            return warp_matrix

        except Exception as e:
            logger.debug(f"ECC refinement skipped: {e}")
            return None

    # ─────────────────────────────────────────
    # MAIN ALIGN — fallback chain
    # ─────────────────────────────────────────
    def align(self, original, modified):
        """
        Align modified image to original using best available method.
        Fallback chain: SIFT → ORB → Phase Correlation → Identity
        Optionally refines with ECC.
        """
        try:
            orig_gray = self._to_gray(original)
            mod_gray  = self._to_gray(modified)
            h, w      = original.shape[:2]

            H      = None
            method = "identity"

            # ── Try SIFT ──
            H = self._align_sift(orig_gray, mod_gray, original.shape)
            if H is not None:
                method = "SIFT"

            # ── Try ORB ──
            if H is None:
                H = self._align_orb(orig_gray, mod_gray, original.shape)
                if H is not None:
                    method = "ORB"

            # ── Try Phase Correlation ──
            if H is None:
                H = self._align_phase_correlation(orig_gray, mod_gray)
                if H is not None:
                    method = "phase_correlation"

            # ── Apply homography ──
            if H is None or method == "identity":
                logger.info("Alignment: identity (no transform needed)")
                return modified

            aligned_gray = cv2.warpPerspective(
                mod_gray, H, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE)

            # ── ECC refinement ──
            ecc = self._refine_ecc(orig_gray, aligned_gray)
            if ecc is not None:
                aligned_gray = cv2.warpAffine(
                    aligned_gray, ecc, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
                method += "+ECC"

            logger.info(f"Alignment method: {method}")

            # Return in same format as input
            if len(modified.shape) == 3:
                aligned = cv2.warpPerspective(
                    modified, H, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
                if ecc is not None:
                    aligned = cv2.warpAffine(
                        aligned, ecc, (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
                return aligned

            return aligned_gray

        except Exception as e:
            logger.error(f"Alignment error: {e}")
            import traceback; traceback.print_exc()
            return modified