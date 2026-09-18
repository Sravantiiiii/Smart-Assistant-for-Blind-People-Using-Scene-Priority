import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- LOAD CALIBRATION ----------------
data = np.load("stereo_params.npz")
mtxL, distL = data["mtxL"], data["distL"]
mtxR, distR = data["mtxR"], data["distR"]
R, T = data["R"], data["T"]

# ---------------- CAMERAS ----------------
capL = cv2.VideoCapture("left.mp4")
capR = cv2.VideoCapture("right.mp4")

# ---------------- YOLO11 ----------------
model = YOLO("yolo11m.pt")
names = model.model.names

# ---------------- STEREO MATCHER ----------------
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=7,
    P1=8 * 3 * 7**2,
    P2=32 * 3 * 7**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# ---------------- FOCAL & BASELINE ----------------
focal_length = mtxL[0, 0]
baseline = abs(T[0])   # meters

print("Press Q to quit")

while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    # -------- DISPARITY --------
    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = 0.1

    # -------- DEPTH MAP (METERS) --------
    depth = (focal_length * baseline) / disp

    # -------- YOLO DETECTION --------
    results = model(frameL, conf=0.4, imgsz=640)
    det = results[0]

    if det.boxes is not None:
        boxes = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # get depth at object center
            d = depth[cy, cx]
            d = round(float(d), 2)

            label = f"{names[int(cls)]} | {d} m"

            cv2.rectangle(frameL, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frameL, label, (x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Stereo Depth + YOLO11", frameL)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
