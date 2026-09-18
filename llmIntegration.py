import cv2
import torch
import time
import queue
import threading
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
import sys, os
sys.path.append("Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2
import pyttsx3

# ================= CONFIG =================
CONF_THRES = 0.5
DEPTH_SCALE = 120.0
HISTORY_LEN = 6
SPEAK_EVERY = 2.0
DEPTH_EVERY_N_FRAMES = 3
COLLECT_WINDOW = 3.0

# ================= DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ================= SAFE TTS =================
tts_engine = pyttsx3.init()
speech_queue = queue.Queue()
tts_busy = threading.Event()

def tts_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        tts_busy.set()
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_busy.clear()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# ================= YOLO =================
model = YOLO("yolo11s.pt")
model.to(device)
class_names = model.model.names

# ================= DEPTH MODEL =================
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

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not accessible")

# ================= TRACK STORAGE =================
track_distances = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
track_instance_names = {}

# ================= 3-SECOND COLLECTION =================
collected_data = defaultdict(list)
collection_start = time.time()

last_depth = None
frame_id = 0
last_speak_time = 0
last_speech_text = None

# ================= DEPTH VISUALIZATION =================
def visualize_depth(depth_gray):
    return cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    frame_id += 1

    # ---------- YOLO ----------
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRES,
        imgsz=416,
        tracker="bytetrack.yaml",
        device=device,
        verbose=False
    )

    det = results[0]
    if det.boxes is None or det.boxes.id is None:
        cv2.imshow("Blind Guidance System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    boxes = det.boxes.xyxy.cpu().numpy()
    classes = det.boxes.cls.cpu().numpy()
    ids = det.boxes.id.cpu().numpy().astype(int)

    # ---------- DEPTH ----------
    if frame_id % DEPTH_EVERY_N_FRAMES == 0 or last_depth is None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inp = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            d = depth_model(inp)

        d = d.squeeze().cpu().numpy()
        d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        last_depth = cv2.resize(d, (w, h))

    depth = last_depth
    active_ids = set()

    # ---------- PROCESS OBJECTS ----------
    for box, cls, tid in zip(boxes, classes, ids):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        depth_val = depth[cy, cx]
        if depth_val == 0:
            continue

        distance = round(DEPTH_SCALE / depth_val, 2)
        base_name = class_names[int(cls)]
        instance_name = f"{base_name}{tid}"

        track_distances[tid].append(distance)
        track_instance_names[tid] = instance_name
        active_ids.add(tid)

        collected_data[instance_name].append(distance)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{instance_name} {distance}m",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # ---------- CLEAN LOST ----------
    for tid in list(track_distances.keys()):
        if tid not in active_ids:
            del track_distances[tid]
            track_instance_names.pop(tid, None)

    # ---------- PRINT 3-SECOND DATA (SORTED) ----------
    now = time.time()
    if now - collection_start >= COLLECT_WINDOW and collected_data:

        print("\n--- COLLECTED DATA (3 seconds) ---")
        sorted_data = []
        for obj, dists in collected_data.items():
            avg_dist = round(sum(dists) / len(dists), 2)
            sorted_data.append((avg_dist, obj, len(dists)))

        sorted_data.sort(key=lambda x: x[0])

        for avg_dist, obj, samples in sorted_data:
            print(f"{obj}: {avg_dist} meters ({samples} samples)")
        print("--------------------------------\n")

        collected_data.clear()
        collection_start = now

    # ---------- SPEECH (SORTED) ----------
    if now - last_speak_time >= SPEAK_EVERY and not tts_busy.is_set():

        if track_distances:
            objs = []
            for tid, dists in track_distances.items():
                avg = sum(dists) / len(dists)
                objs.append((avg, track_instance_names[tid]))

            objs.sort(key=lambda x: x[0])

            last_speech_text = ", ".join(
                f"{name} at {round(dist)} meter{'s' if round(dist)!=1 else ''}"
                for dist, name in objs
            )

        if last_speech_text:
            speak(last_speech_text)
            last_speak_time = now

    # ---------- DISPLAY ----------
    if last_depth is not None:
        depth_colored = visualize_depth(last_depth)
        cv2.imshow("Depth Camera", depth_colored)

    cv2.imshow("Blind Guidance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
speech_queue.put(None)
