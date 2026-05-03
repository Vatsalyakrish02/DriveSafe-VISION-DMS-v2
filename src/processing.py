import cv2
import numpy as np


# --- UTILS ---
def sharpen(img):
    kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
    return cv2.filter2D(img, -1, kernel)


def prepare_roi_square(img, pts, pad):
    x, y = [p.x for p in pts], [p.y for p in pts]
    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + pad)

    nx1 = max(0, int(center_x - side / 2))
    ny1 = max(0, int(center_y - side / 2))
    nx2 = min(img.shape[1], int(center_x + side / 2))
    ny2 = min(img.shape[0], int(center_y + side / 2))

    roi = img[ny1:ny2, nx1:nx2]
    if roi.size == 0: return None

    h, w = roi.shape[:2]
    if w < 80:
        scale = 80 / w
        roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    roi = sharpen(roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_LINEAR) / 255.0
    return np.expand_dims(roi.astype(np.float32), axis=0)
