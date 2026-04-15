import fitz
import cv2
import numpy as np

DPI = 150
DIFF_THRESHOLD = 30
MIN_BLOB_AREA = 2000
INTENSITY_GAP = 15
TITLE_BLOCK_CROP = 0.85   # keep top 85%, discard bottom 15% (title block)
BOX_COLOR = (0, 255, 0)   # green
BOX_THICKNESS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL = "ADDED"


def render_page(pdf_path, dpi=DPI):
    doc = fitz.open(pdf_path)
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    doc.close()
    return img


def crop_live_zone(img):
    # Removing bottom 15% cut to allow detection of bottom revision headers
    return img


def find_added_boxes(v1_crop, v2_crop):
    page_w = v2_crop.shape[1]
    
    # Step 1 — pixel diff
    diff = cv2.absdiff(v1_crop, v2_crop)

    # Step 2 — threshold noise
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Step 3 — morphology clean
    k_open  = np.ones((3, 3),   np.uint8)
    # Refined clustering: 50pt vertical (104px) / 20pt horizontal (42px)
    k_close = np.ones((104, 42), np.uint8) 
    cleaned = cv2.morphologyEx(thresh,  cv2.MORPH_OPEN,  k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    # Step 4 — find contours
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    added = []
    gap_limit = int(30 * (DPI / 72.0)) # 30pt gap for splitting
    pad = int(10 * (DPI / 72.0))       # 10pt padding

    page_h = v2_crop.shape[0]
    
    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_BLOB_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # RULE: Aggressive Shrink for bottom-area detections (>80% page height)
        # This targets the bottom annotation to prune title block line intersections.
        if y > 0.8 * page_h and w > 500:
            roi_thresh = thresh[y:y+h, x:x+w]
            proj = np.sum(roi_thresh > 0, axis=0)
            
            # Find all continuous segments with data
            segments = []
            start = -1
            for i in range(len(proj)):
                if proj[i] > 0 and start == -1:
                    start = i
                elif proj[i] == 0 and start != -1:
                    segments.append((start, i-1))
                    start = -1
            if start != -1:
                segments.append((start, len(proj)-1))
            
            # Find segment with max density
            best_s = None
            max_dens = -1
            for s1, s2 in segments:
                count = np.sum(proj[s1:s2+1])
                if count > max_dens:
                    max_dens = count
                    best_s = (s1, s2)
            
            if best_s:
                # Shrink x and w to just this dense segment
                x = x + best_s[0]
                w = best_s[1] - best_s[0] + 1

        # RULE: Aggressive Height Expansion for lower ADDED regions
        if y > 500:
            scan_x1 = max(0, x - 100)
            scan_x2 = min(page_w, x + w + 100)
            roi_full_h = cleaned[y:, scan_x1:scan_x2]
            v_proj = np.sum(roi_full_h > 0, axis=1)
            
            gap_tolerance = int(50 * (DPI / 72.0)) 
            last_valid_row = h
            current_gap = 0
            
            for i in range(len(v_proj)):
                if v_proj[i] > 0:
                    last_valid_row = i + 1
                    current_gap = 0
                else:
                    current_gap += 1
                    if current_gap > gap_tolerance:
                        break
            h = last_valid_row

        # RULE: Horizontal Gap Splitting
        # If the region is wide and has a large internal gap, split it
        # This separates the NOTE from the 3D-View in Diagram 40
        roi_thresh_h = thresh[y:y+h, x:x+w]
        h_proj = np.sum(roi_thresh_h > 0, axis=0)
        
        horiz_gap_limit = int(40 * (DPI / 72.0)) # approx 83px
        sub_segments = []
        seg_start = -1
        current_horiz_gap = 0
        
        for i in range(len(h_proj)):
            if h_proj[i] > 0:
                if seg_start == -1: seg_start = i
                current_horiz_gap = 0
            else:
                if seg_start != -1:
                    current_horiz_gap += 1
                    if current_horiz_gap > horiz_gap_limit:
                        sub_segments.append((seg_start, i - current_horiz_gap))
                        seg_start = -1
        if seg_start != -1:
            sub_segments.append((seg_start, len(h_proj)-1))

        # If we found multiple segments, add them as separate boxes
        if len(sub_segments) > 1:
            for s1, s2 in sub_segments:
                sub_w = s2 - s1 + 1
                sub_x = x + s1
                # Recalculate tight y/h for this sub-segment
                sub_roi = thresh[y:y+h, sub_x:sub_x+sub_w]
                v_proj_sub = np.sum(sub_roi > 0, axis=1)
                valid_y = np.where(v_proj_sub > 0)[0]
                if len(valid_y) > 0:
                    sub_y_off = valid_y[0]
                    sub_h_new = valid_y[-1] - sub_y_off + 1
                    
                    # Apply 10pt padding
                    pad = int(10 * (DPI / 72.0))
                    added.append((max(0, sub_x - pad), max(0, y+sub_y_off - pad), sub_w + 2*pad, sub_h_new + 2*pad))
            continue # Skip the main box as we added sub-boxes instead
            
        # Apply 10pt padding for non-split boxes
        pad = int(10 * (DPI / 72.0))
        x_p = max(0, x - pad)
        y_p = max(0, y - pad)
        w_p = w + 2 * pad
        h_p = h + 2 * pad

        # Step 5 — classify ADDED: V2 darker (more ink) than V1
        v1_region = v1_crop[y:y+h, x:x+w].astype(float)
        v2_region = v2_crop[y:y+h, x:x+w].astype(float)

        if (v1_region.mean() - v2_region.mean()) > INTENSITY_GAP:
            added.append((x_p, y_p, w_p, h_p))

    return added


def build_output(v1_img, v2_img, added_boxes, output_path):

    def to_bgr(gray):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # V1 and V2 panels — pristine, no modifications
    v1_out   = to_bgr(v1_img.copy())
    v2_out   = to_bgr(v2_img.copy())

    # analysis panel — V2 as base, boxes drawn on top
    analysis = to_bgr(v2_img.copy())

    GREEN = (0, 200, 0)
    THICKNESS = 4

    # scale boxes back to full page coordinates
    # (boxes were found on cropped working image,
    #  display image is full page — no rescaling needed
    #  if crop only removes bottom, x coords are identical,
    #  y coords are identical since we crop from bottom)
    for (x, y, w, h) in added_boxes:
        cv2.rectangle(analysis, (x, y), (x+w, y+h), GREEN, THICKNESS)
        cv2.putText(analysis, "ADDED",
                    (x + 10, y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, GREEN, 3)

    # resize all three panels to same height for side-by-side
    target_h = 1000
    def fit(img):
        r = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), target_h),
                          interpolation=cv2.INTER_AREA)

    p1 = fit(v1_out)
    p2 = fit(v2_out)
    p3 = fit(analysis)

    panel = np.hstack([p1, p2, p3])

    # header labels on white bar at top
    bar = np.ones((60, panel.shape[1], 3), dtype=np.uint8) * 255
    labels = ["V1 (ORIGINAL)", "V2 (REVISION)",
              f"ANALYSIS | ADDED: {len(added_boxes)}"]
    for i, txt in enumerate(labels):
        cv2.putText(bar, txt,
                    (i * p1.shape[1] + 15, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

    final = np.vstack([bar, panel])
    cv2.imwrite(output_path, final)

    print(f"Saved: {output_path}")
    print(f"ADDED boxes found: {len(added_boxes)}")
    for i, (x, y, w, h) in enumerate(added_boxes):
        print(f"  Box {i+1}: x={x} y={y} w={w} h={h}")


def run(v1_pdf, v2_pdf, output_png="visuals/pass1_added_output.png"):
    import os
    os.makedirs("visuals", exist_ok=True)

    # SET A — pristine display copies
    v1_display = render_page(v1_pdf)
    v2_display = render_page(v2_pdf)

    # SET B — working copies for diff
    v1_work = crop_live_zone(v1_display.copy())
    v2_work = crop_live_zone(v2_display.copy())

    # find boxes using working copies
    added_boxes = find_added_boxes(v1_work, v2_work)

    # build output using pristine display copies
    build_output(v1_display, v2_display, added_boxes, output_png)


if __name__ == "__main__":
    import sys
    # Usage: python pass1_added.py v1.pdf v2.pdf
    run(sys.argv[1], sys.argv[2])
