import cv2
import numpy as np
import glob
import sys

# ---------------- CONFIG ----------------
CHECKERBOARD = (9, 6)
square_size = 0.025  # meters

# ---------------- OBJECT POINTS ----------------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints_l = []
imgpoints_r = []

# ---------------- LOAD IMAGES ----------------
left_imgs = sorted(glob.glob("calib/left/*.jpg"))
right_imgs = sorted(glob.glob("calib/right/*.jpg"))

if len(left_imgs) == 0 or len(right_imgs) == 0:
    print("❌ No calibration images found.")
    print("Make sure you have:")
    print("  calib/left/*.jpg")
    print("  calib/right/*.jpg")
    sys.exit(1)

# ---------------- FIND CORNERS ----------------
img_size = None

for l_img, r_img in zip(left_imgs, right_imgs):
    imgL = cv2.imread(l_img)
    imgR = cv2.imread(r_img)

    if imgL is None or imgR is None:
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    if img_size is None:
        img_size = grayL.shape[::-1]

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD)

    if retL and retR:
        objpoints.append(objp)
        imgpoints_l.append(cornersL)
        imgpoints_r.append(cornersR)

# ---------------- VALIDATION ----------------
if len(objpoints) < 5:
    print("❌ Not enough valid stereo pairs found.")
    print("Take at least 10–15 good chessboard images for each camera.")
    sys.exit(1)

# ---------------- CALIBRATE EACH CAMERA ----------------
retL, mtxL, distL, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_l, img_size, None, None
)

retR, mtxR, distR, _, _ = cv2.calibrateCamera(
    objpoints, imgpoints_r, img_size, None, None
)

# ---------------- STEREO CALIBRATION ----------------
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_FIX_INTRINSIC

ret, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    mtxL, distL, mtxR, distR,
    img_size,
    criteria=criteria,
    flags=flags
)

# ---------------- SAVE ----------------
np.savez(
    "stereo_params.npz",
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R=R, T=T
)

print("✅ Stereo calibration completed.")
print("Saved to stereo_params.npz")
