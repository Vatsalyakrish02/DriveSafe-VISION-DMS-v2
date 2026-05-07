import os
import dlib


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "model"))
DLIB_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PATH)


def clamp_rect(rect, w, h):
    l = max(0, rect.left())
    t = max(0, rect.top())
    r = min(w - 1, rect.right())
    b = min(h - 1, rect.bottom())
    return dlib.rectangle(l, t, r, b)

# DriveSafe-VISION-DMS-v2
# Developed by Vatsalya
# GitHub: https://github.com/Vatsalyakrish02
# License: MIT