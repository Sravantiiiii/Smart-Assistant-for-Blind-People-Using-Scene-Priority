

import sys
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

# -------------------------------------------------
# ADD DEPTH ANYTHING PATH
# -------------------------------------------------
sys.path.append("Depth-Anything-V2")
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEPTH_SCALE = 120.0      # calibrate later
CONF_THRES = 0.4

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------------------------
# LOAD YOLO11
# -------------------------------------------------
model = YOLO("yolo11x.pt")     # yolo11s.pt for faster
class_names = model.model.names

# -------------------------------------------------
# LOAD DEPTH MODEL
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

# -------------------------------------------------
# DEPTH TRANSFORM
# -------------------------------------------------
transform = Compose([
    Resize((518, 518)),
    ToTensor(),
    Normalize(mean=0.5, std=0.5),
])

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

    # ---------------- YOLO + BYTETRACK ----------------
    results = model.track(
        source=frame,
        persist=True,
        conf=CONF_THRES,
        imgsz=640,
        tracker="bytetrack.yaml",
        verbose=False
    )

    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes else []
    classes = det.boxes.cls.cpu().numpy() if det.boxes else []
    track_ids = det.boxes.id.cpu().numpy() if det.boxes.id is not None else []

    # ---------------- DEPTH ESTIMATION ----------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_model(input_tensor)

    depth = depth.squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = depth.astype(np.uint8)

    # Resize depth to frame size
    depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))

    # ---------------- DRAW + DISTANCE ----------------
    for box, cls, tid in zip(boxes, classes, track_ids):
        x1, y1, x2, y2 = map(int, box)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)

        region = depth[y1:y2, x1:x2]
        if region.size == 0:
            continue

        mean_depth = np.median(region)
        if mean_depth < 1:
            continue

        # Approx distance in meters
        distance = round(DEPTH_SCALE / mean_depth, 2)

        name = class_names[int(cls)]
        label = f"{name} #{int(tid)} | {distance} m"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ---------------- SHOW ----------------
    cv2.imshow("YOLO11 + ByteTrack + Depth", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
