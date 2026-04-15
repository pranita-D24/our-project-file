import fitz, cv2, numpy as np

def render_page(pdf_path, page_num=0, dpi=150):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w)
    return img

v1 = render_page(r"c:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139.PDF")
v2 = render_page(r"c:\Trivim Internship\engineering_comparison_system\Drawings\PRV73B124139 - Copy.PDF")

def crop_live_zone(img):
    h, w = img.shape
    return img[0:int(h * 0.85), 0:w]

v1c = crop_live_zone(v1)
v2c = crop_live_zone(v2)

diff = cv2.absdiff(v1c, v2c)

_, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

kernel_open  = np.ones((3,3),  np.uint8)
kernel_close = np.ones((60,60), np.uint8)

cleaned = cv2.morphologyEx(thresh,  cv2.MORPH_OPEN,  kernel_open)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

MIN_AREA = 500
boxes = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    boxes.append((x, y, w, h))

added_boxes = []

for (x, y, w, h) in boxes:
    v1_region = v1c[y:y+h, x:x+w].astype(float)
    v2_region = v2c[y:y+h, x:x+w].astype(float)
    
    v1_mean = v1_region.mean()
    v2_mean = v2_region.mean()
    
    # V2 darker = new ink added (lower value = darker in grayscale)
    if (v1_mean - v2_mean) > 60:
        added_boxes.append((x, y, w, h))

def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

v1_out  = to_bgr(v1c)
v2_out  = to_bgr(v2c)
analysis = to_bgr(v2c)

GREEN = (0, 255, 0)
THICKNESS = 6

for (x, y, w, h) in added_boxes:
    cv2.rectangle(analysis, (x, y), (x+w, y+h), GREEN, THICKNESS)
    cv2.putText(analysis, "ADDED", (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)

target_h = 900
def resize_to_height(img, h):
    ratio = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1]*ratio), h))

p1 = resize_to_height(v1_out,  target_h)
p2 = resize_to_height(v2_out,  target_h)
p3 = resize_to_height(analysis, target_h)

panel = np.hstack([p1, p2, p3])

for i, label in enumerate(["V1 (ORIGINAL)", "V2 (REVISION)", "ANALYSIS"]):
    x_offset = i * p1.shape[1] + 10
    cv2.putText(panel, label, (x_offset, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,0), 3)

output_path = r"c:\Trivim Internship\engineering_comparison_system\raster_diff_v2_output.png"
cv2.imwrite(output_path, panel)
print(f"ADDED boxes found: {len(added_boxes)}")
for i, box in enumerate(added_boxes):
    print(f" Box {i+1}: {box}")
