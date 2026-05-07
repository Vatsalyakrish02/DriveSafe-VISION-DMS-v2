import os
import tensorflow as tf
import tensorflow_hub as hub


# =========================
# GPU CONFIG
# =========================
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return []

    memory_limit = os.environ.get("TF_GPU_MEMORY_LIMIT_MB")

    try:
        if memory_limit:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=int(memory_limit))]
            )
        else:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    return gpus


gpus = configure_gpu()


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(PROJECT_ROOT, "model"))
EYE_MODEL_PATH = os.path.join(MODEL_DIR, "20260506-1628-eye-expert-full-mobilenetv2-Adam.h5")
YAWN_MODEL_PATH = os.path.join(MODEL_DIR, "20260419-1313-yawn-expert-full-mobilenetv2-Adam.h5")


# =========================
# LOAD MODELS
# =========================
custom_objs = {'KerasLayer': hub.KerasLayer}
device_name = '/GPU:0' if gpus else '/CPU:0'

with tf.device(device_name):
    eye_model = tf.keras.models.load_model(
        EYE_MODEL_PATH,
        custom_objects=custom_objs,
        compile=False
    )
    yawn_model = tf.keras.models.load_model(
        YAWN_MODEL_PATH,
        custom_objects=custom_objs,
        compile=False
    )

# DriveSafe-VISION-DMS-v2
# Developed by Vatsalya
# GitHub: https://github.com/Vatsalyakrish02
# License: MIT