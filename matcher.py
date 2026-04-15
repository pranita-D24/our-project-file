# matcher.py — v3.0
# Key fixes:
# - Shape type mismatch no longer instantly returns 0
#   (a rectangle that grows can become "complex" — same object)
# - Better area scoring with soft falloff instead of hard reject
# - More reliable MOVED vs MODIFIED distinction
# - Cleaner relocated pair detection

import cv2
import numpy as np
import logging
from scipy.spatial.distance import cosine
from config import (
    AREA_TOLERANCE,
    SHAPE_SIMILARITY_THRESH
)

logger = logging.getLogger(__name__)


class ObjectMatcher:

    def __init__(self):
        self.orb      = cv2.ORB_create(1000)
        self.bf       = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.bf_patch = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        logger.info("ObjectMatcher v3.0 initialized")

    # ══════════════════════════════════════
    # SIMILARITY PRIMITIVES
    # ══════════════════════════════════════
    def _hu_similarity(self, hu1, hu2):
        try:
            def log_hu(hu):
                return np.array([
                    -np.sign(h) * np.log10(abs(h) + 1e-10)
                    for h in hu])
            sim = 1.0 - cosine(log_hu(hu1), log_hu(hu2))
            return max(0.0, float(sim))
        except Exception:
            return 0.0

    def _contour_similarity(self, cnt1, cnt2):
        try:
            d = cv2.matchShapes(cnt1, cnt2,
                                cv2.CONTOURS_MATCH_I1, 0.0)
            return 1.0 / (1.0 + float(d))
        except Exception:
            return 0.0

    def _signature_similarity(self, s1, s2):
        if s1 is None or s2 is None:
            return 0.0
        try:
            v1 = np.asarray(s1, dtype=np.float32)
            v2 = np.asarray(s2, dtype=np.float32)
            d  = np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6
            return float(max(0.0, np.dot(v1, v2) / d))
        except Exception:
            return 0.0

    def _patch_similarity(self, p1, p2):
        return self._signature_similarity(p1, p2)

    def _orb_desc_similarity(self, d1, d2):
        if d1 is None or d2 is None:
            return 0.0
        try:
            if len(d1) == 0 or len(d2) == 0:
                return 0.0
            matches = self.bf_patch.knnMatch(d1, d2, k=2)
            good = sum(
                1 for pair in matches
                if len(pair) == 2 and
                pair[0].distance < 0.75 * pair[1].distance)
            denom = float(max(1, min(len(d1), len(d2))))
            return float(max(0.0, min(1.0, good / denom)))
        except Exception:
            return 0.0

    # ══════════════════════════════════════
    # AREA / ASPECT
    # ══════════════════════════════════════
    def _area_score(self, area1, area2):
        """
        Soft area score instead of hard reject.
        Returns 0→1 where 1 = identical area.
        Falls off smoothly rather than cliff at AREA_TOLERANCE.
        """
        if area1 == 0 or area2 == 0:
            return 0.0
        ratio = min(area1, area2) / max(area1, area2)
        return float(ratio)

    def _area_similar(self, area1, area2):
        return self._area_score(area1, area2) >= (1.0 - AREA_TOLERANCE)

    def _aspect_similar(self, bbox1, bbox2, tolerance=0.28):
        _, _, w1, h1 = bbox1
        _, _, w2, h2 = bbox2
        if h1 == 0 or h2 == 0:
            return False
        ar1   = w1 / (h1 + 1e-6)
        ar2   = w2 / (h2 + 1e-6)
        ratio = min(ar1, ar2) / (max(ar1, ar2) + 1e-6)
        return ratio >= (1.0 - tolerance)

    def _shape_type_compatible(self, t1, t2):
        """
        FIX: Shape types don't have to match exactly.
        A rectangle that grows larger or changes proportions
        may be classified as 'complex' by approxPolyDP.
        We allow: same type, OR either side is 'complex'.
        Also allow: square ↔ rectangle (proportional change).
        """
        if t1 == t2:
            return True
        if "complex" in (t1, t2):
            return True
        # Allow square ↔ rectangle (just resized)
        if set((t1, t2)) <= {"square", "rectangle"}:
            return True
        return False

    # ══════════════════════════════════════
    # PRIMARY MATCH SCORE
    # ══════════════════════════════════════
    def _object_similarity(self, obj1, obj2):
        """
        Full similarity score. Position-independent.
        Shape type mismatch now soft-penalizes instead of hard-reject.
        """
        t1 = obj1["shape_type"]
        t2 = obj2["shape_type"]

        # Shape type compatibility (soft)
        type_compat = 1.0 if self._shape_type_compatible(t1, t2) else 0.3

        area_sc  = self._area_score(obj1["area"], obj2["area"])

        # Hard reject if areas are wildly different (>3× ratio)
        if area_sc < 0.33:
            return 0.0

        hu_sc    = self._hu_similarity(
            obj1["hu_moments"], obj2["hu_moments"])
        cnt_sc   = self._contour_similarity(
            obj1["contour"], obj2["contour"])
        sig_sc   = self._signature_similarity(
            obj1.get("shape_signature"),
            obj2.get("shape_signature"))
        patch_sc = self._patch_similarity(
            obj1.get("patch_signature"),
            obj2.get("patch_signature"))
        orb_sc   = self._orb_desc_similarity(
            obj1.get("patch_orb"),
            obj2.get("patch_orb"))

        ar_ok    = self._aspect_similar(obj1["bbox"], obj2["bbox"])
        ar_sc    = 1.0 if ar_ok else 0.4

        raw = (0.25 * hu_sc   +
               0.18 * area_sc +
               0.07 * ar_sc   +
               0.18 * cnt_sc  +
               0.17 * sig_sc  +
               0.10 * patch_sc+
               0.05 * orb_sc)

        return float(raw * type_compat)

    # ══════════════════════════════════════
    # FALLBACK SCORE — relaxed
    # ══════════════════════════════════════
    def _fallback_similarity(self, obj1, obj2):
        t1 = obj1["shape_type"]
        t2 = obj2["shape_type"]

        if not self._shape_type_compatible(t1, t2):
            return 0.0

        area_sc = self._area_score(obj1["area"], obj2["area"])
        if area_sc < 0.45:
            return 0.0

        hu_sc    = self._hu_similarity(
            obj1["hu_moments"], obj2["hu_moments"])
        cnt_sc   = self._contour_similarity(
            obj1["contour"], obj2["contour"])
        sig_sc   = self._signature_similarity(
            obj1.get("shape_signature"),
            obj2.get("shape_signature"))
        patch_sc = self._patch_similarity(
            obj1.get("patch_signature"),
            obj2.get("patch_signature"))
        orb_sc   = self._orb_desc_similarity(
            obj1.get("patch_orb"),
            obj2.get("patch_orb"))

        ar_ok    = self._aspect_similar(obj1["bbox"], obj2["bbox"],
                                        tolerance=0.35)
        ar_sc    = 1.0 if ar_ok else 0.4

        return float(
            0.22 * hu_sc    +
            0.20 * cnt_sc   +
            0.20 * sig_sc   +
            0.15 * patch_sc +
            0.12 * orb_sc   +
            0.08 * area_sc  +
            0.03 * ar_sc)

    # ══════════════════════════════════════
    # RELOCATION SCORE (moved, same shape)
    # ══════════════════════════════════════
    def _pure_shape_reloc_score(self, o1, o2):
        """
        For relocated objects: area must be very similar (moved = same size).
        If area changed >30%, it's a resize not a move.
        """
        a1, a2 = o1["area"], o2["area"]
        aratio = min(a1, a2) / (max(a1, a2) + 1e-6)
        if aratio < 0.55:
            return 0.0

        sig   = self._signature_similarity(
            o1.get("shape_signature"), o2.get("shape_signature"))
        hu    = self._hu_similarity(
            o1["hu_moments"], o2["hu_moments"])
        cnt   = self._contour_similarity(
            o1["contour"], o2["contour"])
        orb   = self._orb_desc_similarity(
            o1.get("patch_orb"), o2.get("patch_orb"))
        patch = self._patch_similarity(
            o1.get("patch_signature"), o2.get("patch_signature"))

        if (o1.get("shape_signature") is None or
                o2.get("shape_signature") is None):
            return float(0.30*hu + 0.30*cnt + 0.25*patch + 0.15*orb)

        return float(0.28*sig + 0.22*hu + 0.14*cnt +
                     0.24*patch + 0.12*orb)

    # ══════════════════════════════════════
    # RECONCILE RELOCATED PAIRS
    # ══════════════════════════════════════
    def _reconcile_relocated_pairs(self, removed, added, matched):
        if not removed or not added:
            return removed, added

        scored = []
        for ri, r in enumerate(removed):
            for ai, a in enumerate(added):
                s = self._pure_shape_reloc_score(r, a)
                if s > 0 and self._aspect_similar(
                        r["bbox"], a["bbox"], tolerance=0.30):
                    scored.append((s, ri, ai))

        scored.sort(key=lambda x: x[0], reverse=True)

        used_r = set()
        used_a = set()
        RELOC_THRESH = 0.72

        for s, ri, ai in scored:
            if s < RELOC_THRESH:
                break
            if ri in used_r or ai in used_a:
                continue
            r = removed[ri]
            a = added[ai]
            area_chg = round(
                abs(r["area"] - a["area"])
                / (r["area"] + 1e-6) * 100, 1)
            if area_chg > 30.0:
                continue
            used_r.add(ri)
            used_a.add(ai)
            matched.append({
                "v1_object"  : r,
                "v2_object"  : a,
                "similarity" : round(float(s), 3),
                "area_change": area_chg,
            })

        new_removed = [removed[i] for i in range(len(removed))
                       if i not in used_r]
        new_added   = [added[i]   for i in range(len(added))
                       if i not in used_a]
        return new_removed, new_added

    # ══════════════════════════════════════
    # DUPLICATE FRAGMENT FILTER
    # ══════════════════════════════════════
    def _looks_like_duplicate_fragment(self, obj, pool):
        best = max((self._fallback_similarity(obj, other)
                    for other in pool), default=0.0)
        return best >= 0.82

    # ══════════════════════════════════════
    # MOVED VS MODIFIED
    # ══════════════════════════════════════
    def _is_modified(self, obj1, obj2, sim_score):
        """
        Returns True if the object was RESIZED/MODIFIED,
        False if it was simply MOVED (same shape).
        """
        area_chg  = abs(obj1["area"] - obj2["area"]) \
            / (obj1["area"] + 1e-6)
        perim_chg = abs(obj1["perimeter"] - obj2["perimeter"]) \
            / (obj1["perimeter"] + 1e-6)
        sig_sc    = self._signature_similarity(
            obj1.get("shape_signature"),
            obj2.get("shape_signature"))

        # Very similar shape signature → moved
        if sig_sc >= 0.88:
            return False
        # Moderate shape + small area change → moved
        if sig_sc >= 0.80 and area_chg < 0.12:
            return False
        # Shape changed significantly → modified
        if area_chg > 0.20 or perim_chg > 0.20:
            return sig_sc < 0.76
        if sim_score < 0.76:
            return True
        return False

    # ══════════════════════════════════════
    # CORE MATCH ENGINE
    # ══════════════════════════════════════
    def match_objects(self, objects1, objects2):
        """
        4-pass matching:
        Pass 1 — Strict primary score matrix
        Pass 2 — Relaxed fallback
        Pass 3 — Relocated moved objects
        Pass 4 — Fragment noise suppression
        """
        try:
            n = len(objects1)
            m = len(objects2)
            logger.info(f"Matching {n} vs {m} objects...")

            if not objects1 and not objects2:
                return self._empty_result()
            if not objects1:
                return {**self._empty_result(),
                        "added": list(objects2), "added_count": m}
            if not objects2:
                return {**self._empty_result(),
                        "removed": list(objects1), "removed_count": n}

            # ── Pass 1: Primary ──
            sim_matrix = np.zeros((n, m))
            for i, o1 in enumerate(objects1):
                for j, o2 in enumerate(objects2):
                    sim_matrix[i, j] = self._object_similarity(o1, o2)

            matched_1 = set()
            matched_2 = set()
            pairs     = []

            flat = [(sim_matrix[i,j], i, j)
                    for i in range(n)
                    for j in range(m)]
            flat.sort(key=lambda x: x[0], reverse=True)

            for score, i, j in flat:
                if score < SHAPE_SIMILARITY_THRESH:
                    break
                if i in matched_1 or j in matched_2:
                    continue
                pairs.append((i, j, score))
                matched_1.add(i)
                matched_2.add(j)

            # ── Pass 2: Fallback ──
            unmatched_1 = [i for i in range(n) if i not in matched_1]
            unmatched_2 = [j for j in range(m) if j not in matched_2]

            fallback = []
            for i in unmatched_1:
                for j in unmatched_2:
                    s = self._fallback_similarity(objects1[i], objects2[j])
                    fallback.append((s, i, j))

            fallback.sort(key=lambda x: x[0], reverse=True)

            FALLBACK_THRESHOLD = 0.70
            for score, i, j in fallback:
                if score < FALLBACK_THRESHOLD:
                    break
                if i in matched_1 or j in matched_2:
                    continue
                pairs.append((i, j, score))
                matched_1.add(i)
                matched_2.add(j)

            # ── Classify pairs ──
            matched  = []
            modified = []
            for i, j, score in pairs:
                o1 = objects1[i]
                o2 = objects2[j]
                area_chg = round(
                    abs(o1["area"] - o2["area"])
                    / (o1["area"] + 1e-6) * 100, 1)
                if self._is_modified(o1, o2, score):
                    modified.append({
                        "v1_object"  : o1,
                        "v2_object"  : o2,
                        "similarity" : round(score, 3),
                        "area_change": area_chg
                    })
                else:
                    matched.append({
                        "v1_object"  : o1,
                        "v2_object"  : o2,
                        "similarity" : round(score, 3),
                        "area_change": area_chg
                    })

            raw_removed = [objects1[i] for i in range(n)
                           if i not in matched_1]
            raw_added   = [objects2[j] for j in range(m)
                           if j not in matched_2]

            # ── Pass 3: Relocated ──
            raw_removed, raw_added = self._reconcile_relocated_pairs(
                raw_removed, raw_added, matched)

            # ── Pass 4: Fragment suppression ──
            removed = [obj for obj in raw_removed
                       if not self._looks_like_duplicate_fragment(
                           obj, objects2)]
            added   = [obj for obj in raw_added
                       if not self._looks_like_duplicate_fragment(
                           obj, objects1)]

            result = {
                "matched"       : matched,
                "modified"      : modified,
                "added"         : added,
                "removed"       : removed,
                "total_v1"      : n,
                "total_v2"      : m,
                "match_count"   : len(matched),
                "modified_count": len(modified),
                "added_count"   : len(added),
                "removed_count" : len(removed),
            }

            logger.info(
                f"Result — Matched:{len(matched)} "
                f"Modified:{len(modified)} "
                f"Added:{len(added)} Removed:{len(removed)}")

            return result

        except Exception as e:
            logger.error(f"Matching error: {e}")
            import traceback; traceback.print_exc()
            return self._empty_result()

    def _empty_result(self):
        return {
            "matched": [], "modified": [],
            "added":   [], "removed":  [],
            "total_v1": 0, "total_v2": 0,
            "match_count": 0, "modified_count": 0,
            "added_count": 0, "removed_count":  0,
        }