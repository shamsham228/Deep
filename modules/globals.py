# --- FIXED FILE globals.py ---

import os
from typing import List, Dict, Any
from modules.camera_fast import get_frame, get_fps


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = []
simple_map: Dict[str, Any] = {}

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# ------ ✅ FIXED PROCESSOR SETTINGS ------
frame_processors = ["DLC.FACE-SWAPPER"]  # ✅ Correct processor name

# Processing Options
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False
map_faces: bool = False
color_correction: bool = False
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None
execution_providers= ["CPUExecutionProvider"]
execution_threads = 8
headless: bool | None = False
log_level: str = "error"

# ------ ✅ FIX UI TOGGLES TOO ------
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_swapper": True}  # ✅ Correct key

# Face Swapper Specific Options
face_swapper_enabled: bool = True
opacity: float = 1.0
sharpness: float = 0.0

# Mouth Mask Options
mouth_mask: bool = False
show_mouth_mask_box: bool = False
mask_feather_ratio: int = 12
mask_down_size: float = 0.1
mask_size: float = 1.0

# Temporal interpolation
enable_interpolation: bool = True
interpolation_weight: float = 0.0

# ------ ✅ FIXED MODEL PATHS ------
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# DLC may require either of these names
INSWAPPER_MODEL = os.path.join(MODEL_DIR, "inswapper_128.onnx")
INSWAPPER_FP16_MODEL = os.path.join(MODEL_DIR, "inswapper_128_fp16.onnx")

def check_models():
    """Check both FP32 and FP16 swap models"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"📂 Created models folder: {MODEL_DIR}")

    found_any = False

    # Check normal model
    if os.path.exists(INSWAPPER_MODEL):
        size_mb = os.path.getsize(INSWAPPER_MODEL) / (1024 * 1024)
        if size_mb > 50:
            print(f"✅ Swap model OK ({size_mb:.2f}MB): {INSWAPPER_MODEL}")
            found_any = True
        else:
            print(f"⚠ Swap model corrupted/small, replace it: {INSWAPPER_MODEL}")

    # Check FP16 model too
    if os.path.exists(INSWAPPER_FP16_MODEL):
        size_mb = os.path.getsize(INSWAPPER_FP16_MODEL) / (1024 * 1024)
        if size_mb > 50:
            print(f"✅ Swap FP16 model OK ({size_mb:.2f}MB): {INSWAPPER_FP16_MODEL}")
            found_any = True
        else:
            print(f"⚠ Swap FP16 model corrupted/small, replace it: {INSWAPPER_FP16_MODEL}")

    if not found_any:
        print("❌ No valid face swapper model found!\nDownload `inswapper_128.onnx` or `inswapper_128_fp16.onnx` and paste into models folder.")
        return False

    return True

def update_cam_frame() -> bool:
    """Fetch latest camera frame for high FPS pipeline."""
    global latest_cam_frame
    frame = get_frame()

    if frame is None:
        print("⚠ Warning: No frame captured from fast camera!")
        return False

    latest_cam_frame = frame
    return True

# Usage example:
# if not check_models(): exit()

# --- END OF FIXED FILE globals.py ---
