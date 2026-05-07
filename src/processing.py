import cv2
import numpy as np


# =========================
# UTILS
# =========================
def sharpen(img):
    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3.0, -0.5],
        [0, -0.5, 0]
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def prepare_roi_square(img, pts, pad=0.45, out_size=224):
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    side = int(max(bw, bh) * (1.0 + pad))
    if side < 12:
        return None

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(img.shape[1], cx + side // 2)
    ny2 = min(img.shape[0], cy + side // 2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    roi = img[ny1:ny2, nx1:nx2]
    if roi.size == 0:
        return None

    roi = sharpen(roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    roi = roi.astype(np.float32) / 255.0
    return roi
