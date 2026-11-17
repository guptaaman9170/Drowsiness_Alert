"""
=============================================================================
GeeksforGeeks Reference Implementation: Drowsiness Detection using Dlib & OpenCV
Reference: https://www.geeksforgeeks.org/python/python-opencv-drowsiness-detection/
=============================================================================
This script faithfully matches the algorithm presented in the GeeksforGeeks
tutorial using Dlib's 68 facial landmark predictor model.

Prerequisites:
  1. pip install opencv-python dlib scipy imutils pygame
  2. shape_predictor_68_face_landmarks.dat file must be in this directory.
=============================================================================
"""

import os
import sys
import time
import cv2
import numpy as np
from scipy.spatial import distance as dist
from alert_sound import AlertManager

try:
    import imutils
    from imutils import face_utils
except ImportError:
    imutils = None
    face_utils = None

try:
    import dlib
except ImportError:
    dlib = None

# Model file path
PREDICTOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")


def eye_aspect_ratio(eye):
    """
    Computes Eye Aspect Ratio (EAR) as defined in:
    Tereza Soukupova and Jan Cech. Real-Time Eye Blink Detection using Facial Landmarks. 2016.
    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def main():
    if dlib is None:
        print("[ERROR] 'dlib' library is not installed.")
        print("[INFO] On Windows, pre-compiled wheels or Visual C++ tools are required for dlib.")
        print("[TIP] You can use 'python main.py' instead, which uses MediaPipe and works out of the box!")
        return

    if not os.path.exists(PREDICTOR_PATH):
        print(f"[ERROR] Could not find '{PREDICTOR_PATH}'.")
        print("[INFO] Please download 'shape_predictor_68_face_landmarks.dat' from:")
        print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("  extract the .bz2 archive and place the .dat file in this directory.")
        print("[TIP] Alternatively, run 'python main.py' which auto-downloads its model and works immediately!")
        return

    # Initialize Audio Alert
    alert_mgr = AlertManager()

    # Drowsiness detection parameters from GeeksforGeeks tutorial
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 20

    counter = 0
    alarm_on = False

    print("[INFO] Loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)

    # 68-point Dlib landmark eye indices
    # Right eye: 36-41 (indices 36:42), Left eye: 42-47 (indices 42:48)
    (rStart, rEnd) = (36, 42)
    (lStart, lEnd) = (42, 48)

    print("[INFO] Starting video stream...")
    cam_choice = 0
    if sys.stdin.isatty():
        try:
            print("Select Camera index (press Enter for 0): ", end="", flush=True)
            val = input().strip()
            if val.isdigit():
                cam_choice = int(val)
        except (EOFError, KeyboardInterrupt):
            pass

    cap = None
    for backend in ([cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]):
        temp_cap = cv2.VideoCapture(cam_choice + backend)
        if temp_cap.isOpened():
            cap = temp_cap
            break

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_choice}.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours (convex hulls)
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if EAR is below threshold
            if ear < EYE_AR_THRESH:
                counter += 1
                if counter >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_on:
                        alarm_on = True
                        alert_mgr.start_alarm()

                    cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                counter = 0
                if alarm_on:
                    alarm_on = False
                    alert_mgr.stop_alarm()

            cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("GeeksforGeeks Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    alert_mgr.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

