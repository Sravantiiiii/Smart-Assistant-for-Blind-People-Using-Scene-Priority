import sys, time, threading
import cv2
import torch
import numpy as np
from PIL import Image
from collections import defaultdict, deque
import pyttsx3
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

# -------------------------------------------------
# PATH
# -------------------------------------------------
sys.path.append("Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEPTH_SCALE = 120.0        # rough calibration
CONF_THRES = 0.4
DANGER_DISTANCE = 0.8     # virtual danger threshold (relative)
SPEAK_INTERVAL = 3.0      # seconds between voice messages

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# TTS
# -------------------------------------------------
tts_engine = pyttsx3.init()
last_spoken_time = 0

def speak(text):
    def _run():
        tts_engine.say(text)
        tts_engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# -------------------------------------------------
# YOLO11 + BYTETRACK
# -------------------------------------------------
model = YOLO("yolo11s.pt")   # use yolo11s.pt for speed
class_names = model.model.names

# -------------------------------------------------
# DEPTH MODEL
# -------------------------------------------------
depth_model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
)

depth_model.load_state_dict(
    torch.hub.load_state_dict_from_url(
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        map_location=device
    )
)
depth_model = depth_model.to(device).eval()

transform = Compose([
    Resize((518, 518)),
    ToTensor(),
    Normalize(mean=0.5, std=0.5),
])

# -------------------------------------------------
# TRACK HISTORY
# -------------------------------------------------
track_history = defaultdict(lambda: deque(maxlen=6))   # last 6 distances
track_names = {}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def compute_approach_speed(distances):
    if len(distances) < 2:
        return 0.0
    return distances[-2] - distances[-1]

def estimate_time_to_reach(distances, threshold=DANGER_DISTANCE):
    if len(distances) < 2:
        return float("inf")

    speed = distances[-2] - distances[-1]
    if speed <= 0:
        return float("inf")

    current = distances[-1]
    return max((current - threshold) / speed, 0.01)

def generate_guidance(priority_list):
    if not priority_list:
        return None

    top = priority_list[0]
    eta = top["eta"]
    name = top["name"]

    if eta < 1.5:
        return f"Warning. {name} approaching fast. Stop and move aside."
    elif eta < 3.0:
        return f"Caution. {name} moving closer. Slow down."
    else:
        return None

# -------------------------------------------------
# CAMERA
# -------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

print("Press Q to quit")

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # ---------------- YOLO11 + BYTETRACK ----------------
    results = model.track(
        source=frame,
        persist=True,
        conf=CONF_THRES,
        imgsz=640,
        tracker="bytetrack.yaml",
        verbose=False
    )

    det = results[0]
    if det.boxes is None:
        cv2.imshow("Blind Guidance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    ids = det.boxes.id.cpu().numpy() if det.boxes.id is not None else []

    # ---------------- DEPTH ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_model(input_tensor)

    depth = depth.squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)
    depth = cv2.resize(depth, (w, h))

    # ---------------- PROCESS OBJECTS ----------------
    for box, cls, tid in zip(boxes, classes, ids):
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # use center point depth (more stable)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        depth_val = depth[cy, cx]

        if depth_val < 1:
            continue

        distance = round(DEPTH_SCALE / depth_val, 2)

        track_id = int(tid)
        name = class_names[int(cls)]

        track_names[track_id] = name
        track_history[track_id].append(distance)

        # draw
        label = f"{name} #{track_id} | {distance}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # ---------------- PRIORITY ENGINE ----------------
    priority_list = []

    for tid, dists in track_history.items():
        speed = compute_approach_speed(dists)
        eta = estimate_time_to_reach(dists)

        priority_list.append({
            "track_id": tid,
            "name": track_names.get(tid, "object"),
            "speed": speed,
            "eta": eta
        })

    priority_list.sort(key=lambda x: x["eta"])

    # ---------------- GUIDANCE ----------------
    now = time.time()
    if now - last_spoken_time > SPEAK_INTERVAL:
        guidance = generate_guidance(priority_list)
        if guidance:
            print("GUIDANCE:", guidance)
            speak(guidance)
            last_spoken_time = now

    # ---------------- SHOW ----------------
    cv2.imshow("Blind Guidance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
