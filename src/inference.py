import os
import tensorflow as tf
import tensorflow_hub as hub


# --- ADAPTIVE GPU CONFIG ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enables "Memory Growth" so TF only takes what it needs from the VRAM 
            # instead of pre-allocating the entire 4GB at startup.
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"Successfully configured {len(gpus)} GPU(s) with adaptive memory growth.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"GPU Configuration Error: {e}")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
EYE_MODEL_PATH  = os.path.join(MODEL_DIR, "20250104-2045-full-image-set-mobilenetv2-Adam.h5")
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "20260419-1313-yawn-expert-full-mobilenetv2-Adam.h5")

custom_objs = {'KerasLayer': hub.KerasLayer}
with tf.device('/GPU:0'):
    eye_model  = tf.keras.models.load_model(EYE_MODEL_PATH,  custom_objects=custom_objs, compile=False)
    yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH, custom_objects=custom_objs, compile=False)
