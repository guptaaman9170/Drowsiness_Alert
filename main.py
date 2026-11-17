"""
=============================================================================
Driver Drowsiness & Sleep Alert System
End-to-End Real-Time Computer Vision Pipeline using OpenCV & MediaPipe
Reference: GeeksforGeeks Drowsiness Detection & Soukupova/Cech EAR Algorithm
=============================================================================
"""

import os
import sys
import time
import urllib.request
import argparse
import cv2
import numpy as np
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alert_sound import AlertManager
from emergency_notifier import EmergencyNotifier

# Suppress noisy OpenCV camera backend probe warnings
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# Enable unbuffered / line-buffered stdout so logs appear immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Default EAR & MAR thresholds
DEFAULT_EAR_THRESH = 0.23      # Eye Aspect Ratio threshold (eyes closed if < 0.23)
DEFAULT_CONSEC_FRAMES = 22     # Consecutive frames eyes must be closed to trigger alarm (~0.8s)
DEFAULT_YAWN_THRESH = 0.60     # Mouth Aspect Ratio threshold for yawning
DEFAULT_YAWN_FRAMES = 15       # Consecutive frames mouth must be open

# MediaPipe 468/478 Landmark Indices for Eyes & Mouth
# (Matches the standard 6-point eye model used in the Soukupova/Cech EAR paper)
# Left eye (subject's left side / screen right):
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye (subject's right side / screen left):
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Mouth indices for MAR (yawning detection):
# [Left corner, Right corner, Upper lip center, Lower lip center]
MOUTH_INDICES = [61, 291, 13, 14]


def ensure_model_exists():
    """Ensures face_landmarker.task exists, downloads automatically if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Downloading MediaPipe Face Landmarker model from Google...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"[INFO] Download complete: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Could not download model: {e}")
            print(f"[INFO] Please download manually from:\n{MODEL_URL}\nand save as 'face_landmarker.task'.")
            sys.exit(1)


def scan_available_cameras(max_probe=6):
    """
    Scans system for connected cameras and queries real hardware friendly names
    (e.g. 'Camo', 'OBS Virtual Camera', 'Integrated Webcam') via DirectShow.
    """
    device_names = []
    try:
        from pygrabber.dshow_graph import FilterGraph
        device_names = FilterGraph().get_input_devices()
    except Exception:
        device_names = []

    available = []
    probe_limit = max(max_probe, len(device_names))

    for idx in range(probe_limit):
        name = device_names[idx] if idx < len(device_names) else f"Camera Device {idx}"
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                available.append((idx, w, h, name))
                cap.release()
            elif idx < len(device_names):
                # Registered virtual device (like Camo Studio) even if standby
                available.append((idx, 1280, 720, name))
        except Exception:
            pass

    return available


def prompt_camera_permission(detected_cams, requested_cam=None):
    """
    Asks the user for explicit permission and camera selection by friendly name.
    Supports camera indices (0, 1, 2) or IP Webcam / video stream URLs.
    """
    if requested_cam is not None:
        if isinstance(requested_cam, str) and requested_cam.isdigit():
            requested_cam = int(requested_cam)
        print(f"[INFO] Using requested camera source: {requested_cam}\n", flush=True)
        return requested_cam

    print("\n" + "=" * 68)
    print("📷 CAMERA SELECTION & DEVICE PERMISSION")
    print("=" * 68)

    if not detected_cams:
        print("[WARN] No active webcam index responded to initial probe.")
        default_choice = 0
    else:
        print(f"Found {len(detected_cams)} connected video capture device(s):")
        for item in detected_cams:
            idx, w, h, name = item
            tag = " [Recommended]" if ("camo" in name.lower() or idx == 0) else ""
            print(f"   [{idx}] -> {name}  (Resolution: {w}x{h}){tag}")
        default_choice = detected_cams[0][0]

    print("-" * 68)
    print(f"👉 Press [ENTER] to use: [{default_choice}]")
    print("👉 Or type the camera index number (e.g. 0, 1, 2...)")
    print("👉 Or paste an IP Webcam stream URL (e.g. http://192.168.1.5:8080/video)")
    print("-" * 68)

    # If run in non-interactive terminal (pipe/automation), safely default without hanging
    if not sys.stdin.isatty():
        print(f"[INFO] Non-interactive environment detected. Auto-selecting Camera [{default_choice}].\n", flush=True)
        return default_choice

    try:
        user_input = input(f"Which camera do you want to use? [{default_choice}]: ").strip()
        if not user_input:
            chosen = default_choice
        elif user_input.isdigit():
            chosen = int(user_input)
        else:
            chosen = user_input  # URL or file path
    except (EOFError, KeyboardInterrupt):
        chosen = default_choice

    print(f"[INFO] Permission granted for Camera: {chosen}\n", flush=True)
    return chosen


def open_camera_stream(source):
    """
    Intelligently opens camera source by auto-detecting the optimal backend
    (MSMF vs DirectShow). Modern virtual cameras like Camo Studio require MSMF
    to avoid black frames, while standard USB webcams often use DirectShow.
    """
    cap = None
    if isinstance(source, int):
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        
        # Phase 1: Try backends to find one that returns live, non-black video frames
        for backend in backends:
            temp_cap = cv2.VideoCapture(source + backend)
            if temp_cap.isOpened():
                live_video_confirmed = False
                for _ in range(4):
                    ret, test_frame = temp_cap.read()
                    if ret and test_frame is not None and test_frame.std() > 3.0:
                        live_video_confirmed = True
                        break
                if live_video_confirmed:
                    cap = temp_cap
                    b_name = "MSMF" if backend == cv2.CAP_MSMF else ("DSHOW" if backend == cv2.CAP_DSHOW else "ANY")
                    print(f"[INFO] Camera [{source}] successfully connected via {b_name} ({test_frame.shape[1]}x{test_frame.shape[0]}).", flush=True)
                    break
                temp_cap.release()

        # Phase 2: If none returned non-zero frames yet (e.g. phone screen just waking up), pick MSMF/DSHOW
        if cap is None:
            for backend in backends:
                temp_cap = cv2.VideoCapture(source + backend)
                if temp_cap.isOpened():
                    cap = temp_cap
                    b_name = "MSMF" if backend == cv2.CAP_MSMF else ("DSHOW" if backend == cv2.CAP_DSHOW else "ANY")
                    print(f"[INFO] Camera [{source}] opened with {b_name} (waiting for video stream).", flush=True)
                    break
    else:
        cap = cv2.VideoCapture(source)

    return cap


def calculate_ear(eye_points):
    """
    Computes Eye Aspect Ratio (EAR):
    EAR = (|p2 - p6| + |p3 - p5|) / (2.0 * |p1 - p4|)
    """
    p1, p2, p3, p4, p5, p6 = eye_points
    # Vertical euclidean distances
    a = dist.euclidean(p2, p6)
    b = dist.euclidean(p3, p5)
    # Horizontal euclidean distance
    c = dist.euclidean(p1, p4)

    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def calculate_mar(mouth_points):
    """
    Computes Mouth Aspect Ratio (MAR):
    MAR = |upper_lip - lower_lip| / |left_corner - right_corner|
    """
    p_left, p_right, p_top, p_bottom = mouth_points
    vertical = dist.euclidean(p_top, p_bottom)
    horizontal = dist.euclidean(p_left, p_right)

    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def draw_hud(frame, ear, mar, closed_counter, ear_thresh, consec_frames,
             drowsy_count, yawn_count, alarm_active, yawn_active, alert_mgr, fps,
             cam_id=0, alarm_sec=0.0, sos_active=False, sos_email=""):
    """Draws a clean, professional HUD with status badges, meters, camera info, and alert banners."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay card on top-left
    overlay = frame.copy()
    card_w, card_h = 320, 210
    cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Card border
    border_color = (0, 0, 255) if (alarm_active or sos_active) else ((0, 255, 255) if yawn_active else (70, 70, 70))
    cv2.rectangle(frame, (10, 10), (10 + card_w, 10 + card_h), border_color, 2)

    # Title & FPS
    cv2.putText(frame, "DROWSINESS DETECTOR", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (240, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)

    # Divider line
    cv2.line(frame, (20, 42), (320, 42), (80, 80, 80), 1)

    # EAR Display & Meter
    ear_color = (0, 0, 255) if ear < ear_thresh else (0, 255, 0)
    cv2.putText(frame, f"EAR: {ear:0.3f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Threshold: {ear_thresh:0.2f}", (170, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    # EAR Progress bar
    bar_x, bar_y, bar_max_w, bar_h = 20, 75, 290, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (50, 50, 50), -1)
    bar_fill = int(min(1.0, max(0.0, ear / 0.40)) * bar_max_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_fill, bar_y + bar_h), ear_color, -1)

    # Threshold indicator notch on bar
    thresh_x = bar_x + int((ear_thresh / 0.40) * bar_max_w)
    cv2.line(frame, (thresh_x, bar_y - 2), (thresh_x, bar_y + bar_h + 2), (0, 255, 255), 2)

    # Eye Closure Frame Counter / SOS Timer
    if alarm_active:
        cv2.putText(frame, f"Alarm Active: {alarm_sec:0.1f}s / 10.0s (SOS)", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 140, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Closed Frames: {closed_counter} / {consec_frames}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # MAR Display
    mar_color = (0, 165, 255) if yawn_active else (200, 200, 200)
    cv2.putText(frame, f"MAR (Yawn): {mar:0.2f}", (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, mar_color, 1, cv2.LINE_AA)

    # Statistics & Active Camera
    cv2.putText(frame, f"Alerts: {drowsy_count} | Yawns: {yawn_count}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Audio status & Camera switch hotkey reminder
    audio_txt = "[MUTED]" if alert_mgr.muted else "[ON]"
    audio_col = (0, 120, 255) if alert_mgr.muted else (0, 255, 120)
    cv2.putText(frame, f"Cam: {cam_id} | Snd:{audio_txt} (C:cam | M:mute | Q:quit)", (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, audio_col, 1, cv2.LINE_AA)

    # Prominent Status Banner
    if sos_active:
        # Flashing alternating Red & Yellow emergency border
        pulse = int((time.time() * 8) % 2)
        border_col = (0, 0, 255) if pulse == 0 else (0, 215, 255)
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_col, 10)

        # Critical SOS Banner
        banner_w, banner_h = 560, 68
        bx1 = (w - banner_w) // 2
        by1 = 35
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 0, 180), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 215, 255), 2)
        cv2.putText(frame, "CRITICAL SOS: 10s UNRESPONSIVE!", (bx1 + 18, by1 + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Location & SOS Email dispatched to: {sos_email}", (bx1 + 18, by1 + 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    elif alarm_active:
        pulse = int((time.time() * 8) % 2)
        if pulse == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        # Alert banner centered
        banner_w, banner_h = 460, 60
        bx1 = (w - banner_w) // 2
        by1 = 40
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 0, 220), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (255, 255, 255), 2)
        cv2.putText(frame, "DROWSINESS DETECTED! WAKE UP!", (bx1 + 15, by1 + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    elif yawn_active:
        banner_w, banner_h = 360, 45
        bx1 = (w - banner_w) // 2
        by1 = 40
        cv2.rectangle(frame, (bx1, by1), (bx1 + banner_w, by1 + banner_h), (0, 140, 255), -1)
        cv2.putText(frame, "YAWN DETECTED - TAKE A BREAK", (bx1 + 18, by1 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        # Green "AWAKE & ATTENTIVE" badge at bottom
        badge_w, badge_h = 240, 32
        bx1 = (w - badge_w) // 2
        by1 = h - 45
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (0, 140, 0), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (255, 255, 255), 1)
        cv2.putText(frame, "STATUS: ATTENTIVE", (bx1 + 22, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        badge_w, badge_h = 240, 32
        bx1 = (w - badge_w) // 2
        by1 = h - 45
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (0, 140, 0), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (255, 255, 255), 1)
        cv2.putText(frame, "STATUS: ATTENTIVE", (bx1 + 22, by1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness & Sleep Alert System")
    parser.add_argument("--camera", "-c", default=None,
                        help="Camera index (e.g. 0, 1) or stream URL to use without prompt.")
    parser.add_argument("--sos-threshold", "-t", type=float, default=10.0,
                        help="Continuous alarm duration (seconds) before triggering SOS email (default: 10.0s).")
    parser.add_argument("--sos-email", "-e", default=None,
                        help="Emergency recipient email address (default: configured in email_config.json).")
    args = parser.parse_args()

    ensure_model_exists()
    print("[INFO] Initializing Audio Alert Manager...", flush=True)
    alert_mgr = AlertManager()

    emergency_notifier = EmergencyNotifier(recipient_email=args.sos_email)
    sos_email = emergency_notifier.recipient_email
    print(f"[INFO] Initializing Emergency Notifier (Recipient: {sos_email})...", flush=True)

    print("[INFO] Initializing MediaPipe Face Landmarker...", flush=True)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # Scan available cameras and ask user permission
    detected_cams = scan_available_cameras(max_probe=5)
    current_cam_source = prompt_camera_permission(detected_cams, requested_cam=args.camera)

    cap = open_camera_stream(current_cam_source)
    if cap is None or not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {current_cam_source}", flush=True)
        print("[INFO] Please verify your camera is plugged in and permissions are granted.", flush=True)
        return

    # State variables
    ear_thresh = DEFAULT_EAR_THRESH
    consec_frames = DEFAULT_CONSEC_FRAMES
    yawn_thresh = DEFAULT_YAWN_THRESH
    yawn_frames = DEFAULT_YAWN_FRAMES

    # Emergency SOS variables
    sos_threshold_sec = args.sos_threshold
    sos_email = emergency_notifier.recipient_email
    alarm_start_time = None
    continuous_alarm_sec = 0.0
    sos_dispatched = False
    sos_active = False
    sos_banner_timer = 0.0

    closed_counter = 0
    yawn_counter = 0
    drowsy_event_count = 0
    yawn_event_count = 0
    alarm_active = False
    yawn_active = False

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    print("[INFO] Drowsiness Detection System Running!")
    print("=" * 65)
    print("Safety & Emergency Settings:")
    print(f"  • EAR Closure Threshold  : {ear_thresh:0.2f}")
    print(f"  • Continuous Alarm SOS   : {sos_threshold_sec:0.1f} seconds")
    print(f"  • Emergency Email Alert  : {sos_email}")
    print("-" * 65)
    print("Keyboard Controls (click inside camera window):")
    print("  'q' or 'ESC' : Exit application cleanly")
    print("  'c'          : Switch to next connected camera live")
    print("  'm'          : Mute / Unmute beep alarm")
    print("  '+' / '='    : Increase EAR threshold sensitivity")
    print("  '-' / '_'    : Decrease EAR threshold sensitivity")
    print("  'r'          : Reset fatigue & blink statistics")
    print("=" * 65 + "\n", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from webcam.")
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                fps = 15.0 / elapsed if elapsed > 0 else 30.0
                start_time = time.time()

            # Flip horizontally for natural mirror selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Detect if Camo or external device is streaming black frames (phone disconnected)
            if frame.std() < 2.0:
                cv2.putText(frame, "CAMO / PHONE CAM: Waiting for video feed...",
                            (max(20, w // 2 - 240), h // 2 - 15),
                            cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Please ensure phone is unlocked & connected in Camo Studio",
                            (max(20, w // 2 - 260), h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Detect face landmarks
            detection_result = detector.detect(mp_image)

            current_ear = 0.0
            current_mar = 0.0

            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]

                # Extract pixel coordinates for left eye
                left_eye_pts = np.array([
                    [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
                    for idx in LEFT_EYE_INDICES
                ], dtype=np.int32)

                # Extract pixel coordinates for right eye
                right_eye_pts = np.array([
                    [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
                    for idx in RIGHT_EYE_INDICES
                ], dtype=np.int32)

                # Extract pixel coordinates for mouth
                mouth_pts = np.array([
                    [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
                    for idx in MOUTH_INDICES
                ], dtype=np.int32)

                # Calculate EAR for both eyes and average
                left_ear = calculate_ear(left_eye_pts)
                right_ear = calculate_ear(right_eye_pts)
                current_ear = (left_ear + right_ear) / 2.0

                # Calculate MAR
                current_mar = calculate_mar(mouth_pts)

                # Draw eye contours (green if open, red if closed)
                eye_color = (0, 0, 255) if current_ear < ear_thresh else (0, 255, 0)
                cv2.polylines(frame, [left_eye_pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)
                cv2.polylines(frame, [right_eye_pts], isClosed=True, color=eye_color, thickness=1, lineType=cv2.LINE_AA)

                # Draw mouth contour
                mouth_color = (0, 165, 255) if current_mar > yawn_thresh else (200, 200, 0)
                cv2.circle(frame, tuple(mouth_pts[2]), 2, mouth_color, -1)
                cv2.circle(frame, tuple(mouth_pts[3]), 2, mouth_color, -1)
                cv2.line(frame, tuple(mouth_pts[2]), tuple(mouth_pts[3]), mouth_color, 1)

                # Drowsiness Logic (Eyes closed)
                if current_ear < ear_thresh:
                    closed_counter += 1
                    if closed_counter >= consec_frames:
                        if not alarm_active:
                            alarm_active = True
                            alarm_start_time = time.time()
                            sos_dispatched = False
                            drowsy_event_count += 1
                            print(f"[ALERT] Drowsiness Detected! Event #{drowsy_event_count} (EAR={current_ear:.3f})")
                        alert_mgr.start_alarm()

                        # Track continuous alarm duration for Emergency SOS
                        if alarm_start_time is not None:
                            continuous_alarm_sec = time.time() - alarm_start_time
                            if continuous_alarm_sec >= sos_threshold_sec and not sos_dispatched:
                                sos_dispatched = True
                                sos_active = True
                                sos_banner_timer = time.time() + 15.0  # Keep SOS banner visible for 15s
                                print(f"\n[CRITICAL SOS] Driver UNRESPONSIVE for {continuous_alarm_sec:.1f}s!", flush=True)
                                print(f"[INFO] Dispatching emergency notification with location to: {sos_email}...", flush=True)
                                emergency_notifier.dispatch_sos_alert(
                                    continuous_seconds=continuous_alarm_sec,
                                    ear_value=current_ear
                                )
                else:
                    if alarm_active:
                        print(f"[INFO] Eyes opened. Driver attentive again.")
                    closed_counter = 0
                    alarm_active = False
                    alarm_start_time = None
                    continuous_alarm_sec = 0.0
                    alert_mgr.stop_alarm()

                # Yawning Logic
                if current_mar > yawn_thresh:
                    yawn_counter += 1
                    if yawn_counter >= yawn_frames:
                        if not yawn_active:
                            yawn_active = True
                            yawn_event_count += 1
                else:
                    yawn_counter = 0
                    yawn_active = False

            else:
                # No face detected
                cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 120, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 140, 255), 2, cv2.LINE_AA)
                if alarm_active:
                    alert_mgr.stop_alarm()
                    alarm_active = False
                    alarm_start_time = None
                    continuous_alarm_sec = 0.0
                closed_counter = 0

            # Turn off SOS banner after display timer expires
            if sos_banner_timer > 0 and time.time() > sos_banner_timer:
                sos_active = False

            # Render HUD
            draw_hud(frame, current_ear, current_mar, closed_counter, ear_thresh,
                     consec_frames, drowsy_event_count, yawn_event_count,
                     alarm_active, yawn_active, alert_mgr, fps,
                     cam_id=current_cam_source, alarm_sec=continuous_alarm_sec,
                     sos_active=sos_active, sos_email=sos_email)

            # Display output
            cv2.imshow("Driver Drowsiness Alert System", frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[INFO] Exiting on user request...")
                break
            elif key == ord('c') or key == ord('C'):
                # Dynamically switch to next camera
                if isinstance(current_cam_source, int):
                    next_source = (current_cam_source + 1) % 4
                    print(f"\n[INFO] Attempting to switch to Camera [{next_source}]...", flush=True)
                    new_cap = open_camera_stream(next_source)
                    if new_cap is not None:
                        ret_test, _ = new_cap.read()
                        if ret_test:
                            cap.release()
                            cap = new_cap
                            current_cam_source = next_source
                            print(f"[INFO] Successfully switched to Camera [{current_cam_source}]!\n", flush=True)
                        else:
                            new_cap.release()
                            print(f"[WARN] Camera [{next_source}] returned empty frame. Retaining [{current_cam_source}].\n", flush=True)
                    else:
                        print(f"[WARN] Camera [{next_source}] could not be opened. Retaining [{current_cam_source}].\n", flush=True)
            elif key == ord('m') or key == ord('M'):
                is_muted = alert_mgr.toggle_mute()
                print(f"[INFO] Sound {'MUTED' if is_muted else 'UNMUTED'}")
            elif key == ord('+') or key == ord('='):
                ear_thresh = min(0.35, round(ear_thresh + 0.01, 2))
                print(f"[INFO] EAR Threshold increased to: {ear_thresh}")
            elif key == ord('-') or key == ord('_'):
                ear_thresh = max(0.15, round(ear_thresh - 0.01, 2))
                print(f"[INFO] EAR Threshold decreased to: {ear_thresh}")
            elif key == ord('r') or key == ord('R'):
                drowsy_event_count = 0
                yawn_event_count = 0
                closed_counter = 0
                print("[INFO] Statistics reset.")

    except KeyboardInterrupt:
        print("[INFO] Stopped by KeyboardInterrupt.")

    finally:
        # Cleanup
        print("[INFO] Cleaning up resources...")
        alert_mgr.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System terminated cleanly.")


if __name__ == "__main__":
    main()

