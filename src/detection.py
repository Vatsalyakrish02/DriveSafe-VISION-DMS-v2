import os
import dlib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
DLIB_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DLIB_PATH)
