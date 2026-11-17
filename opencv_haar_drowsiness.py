"""
=============================================================================
Pure OpenCV Haar Cascade Drowsiness Detection System
=============================================================================
Zero external ML models required! Uses OpenCV's built-in Haar Cascade XMLs:
  - haarcascade_frontalface_default.xml
  - haarcascade_eye_tree_eyeglasses.xml (or haarcascade_eye.xml)

Logic:
  1. Detects face using Frontal Face Haar Cascade.
  2. Isolates Upper Face Region of Interest (ROI) where eyes are located.
  3. Detects open eyes inside the eye ROI.
  4. If face is detected but NO eyes are found for consecutive frames (eyes closed),
     triggers the Drowsiness Alarm!
=============================================================================
"""

import os
import sys
import time
import cv2
from alert_sound import AlertManager

def main():
    print("[INFO] Starting OpenCV Haar Cascade Drowsiness Detection...")
    alert_mgr = AlertManager()

    # Load Haar Cascades from OpenCV's built-in data folder
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'

    if not os.path.exists(eye_cascade_path):
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    if face_cascade.empty() or eye_cascade.empty():
        print("[ERROR] Failed to load OpenCV Haar Cascades.")
        return

    # Camera selection prompt
    print("\n" + "=" * 60)
    print("📷 CAMERA SELECTION (Haar Cascades)")
    print("=" * 60)
    print("👉 Press [ENTER] for default Camera [0]")
    print("👉 Or type camera index (e.g. 0, 1, 2...)")
    print("-" * 60)
    cam_choice = 0
    if sys.stdin.isatty():
        try:
            val = input("Select Camera index [0]: ").strip()
            if val.isdigit():
                cam_choice = int(val)
        except (EOFError, KeyboardInterrupt):
            pass

    print(f"[INFO] Connecting to camera {cam_choice}...")
    cap = None
    for backend in ([cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]):
        temp_cap = cv2.VideoCapture(cam_choice + backend)
        if temp_cap.isOpened():
            cap = temp_cap
            break

    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open camera {cam_choice}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Threshold for consecutive frames without eyes detected
    EYES_CLOSED_CONSEC_FRAMES = 18
    closed_frames_counter = 0
    alarm_active = False

    print("[INFO] Haar Cascade system running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120)
        )

        face_detected = len(faces) > 0
        eyes_detected = False

        for (fx, fy, fw, fh) in faces:
            # Draw face box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 180, 0), 2)

            # Restrict eye search to upper 55% of the face ROI
            eye_roi_gray = gray[fy:fy + int(fh * 0.55), fx:fx + fw]
            eye_roi_color = frame[fy:fy + int(fh * 0.55), fx:fx + fw]

            eyes = eye_cascade.detectMultiScale(
                eye_roi_gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(25, 25)
            )

            if len(eyes) >= 1:
                eyes_detected = True
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(eye_roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # State determination
        if face_detected:
            if not eyes_detected:
                closed_frames_counter += 1
                if closed_frames_counter >= EYES_CLOSED_CONSEC_FRAMES:
                    if not alarm_active:
                        alarm_active = True
                        alert_mgr.start_alarm()
            else:
                closed_frames_counter = 0
                if alarm_active:
                    alarm_active = False
                    alert_mgr.stop_alarm()
        else:
            closed_frames_counter = 0
            if alarm_active:
                alarm_active = False
                alert_mgr.stop_alarm()

        # Visual status text
        if alarm_active:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(frame, "** DROWSINESS ALERT! WAKE UP! **", (30, 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        elif face_detected and eyes_detected:
            cv2.putText(frame, "STATUS: EYES OPEN (AWAKE)", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif face_detected and not eyes_detected:
            cv2.putText(frame, f"STATUS: EYES CLOSED ({closed_frames_counter}/{EYES_CLOSED_CONSEC_FRAMES})", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "STATUS: NO FACE DETECTED", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

        cv2.imshow("Pure OpenCV Haar Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    alert_mgr.cleanup()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

