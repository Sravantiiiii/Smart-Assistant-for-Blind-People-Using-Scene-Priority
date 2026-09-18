import cv2
import os

os.makedirs("calib/left", exist_ok=True)
os.makedirs("calib/right", exist_ok=True)

capL = cv2.VideoCapture("left.mp4")
capR = cv2.VideoCapture("right.mp4")

i = 0
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        break

    # save every 20th frame (avoid too many similar frames)
    if i % 20 == 0:
        cv2.imwrite(f"calib/left/left_{i}.jpg", frameL)
        cv2.imwrite(f"calib/right/right_{i}.jpg", frameR)

    i += 1

capL.release()
capR.release()
print("Frames extracted.")
