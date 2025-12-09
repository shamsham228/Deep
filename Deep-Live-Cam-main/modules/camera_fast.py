import cv2, time, threading

# Select Fastest backend for Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("⚠ CAP_DSHOW failed, trying CAP_MSMF...")
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

# Set high resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)  # try locking 30 FPS

# Shared frame buffer
frame = None
fps = 0
lock = threading.Lock()  # thread safety

def camera_loop():
    global frame, fps
    prev_time = time.time()

    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            continue

        # Store latest frame safely
        with lock:
            frame = f.copy()

        # Smooth FPS calculation
        curr = time.time()
        diff = curr - prev_time
        if diff > 0:
            new_fps = 1 / diff
            fps = (fps * 0.9) + (new_fps * 0.1)  # stronger smoothing
        prev_time = curr

# Start fast capture thread
threading.Thread(target=camera_loop, daemon=True).start()

def get_frame():
    with lock:
        return frame

def get_fps():
    return round(fps, 30)
