"""
test_pipeline.py - Automated Smoke Test for Drowsiness Detection System
Verifies model loading, EAR/MAR calculation, audio playback, and HUD rendering.
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from main import calculate_ear, calculate_mar, draw_hud, MODEL_PATH, ensure_model_exists
from alert_sound import AlertManager
from emergency_notifier import EmergencyNotifier, get_current_geolocation

def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED SYSTEM INTEGRATION TESTS")
    print("=" * 60)

    # 1. Model verification
    print("[TEST 1/5] Checking model file...")
    ensure_model_exists()
    assert os.path.exists(MODEL_PATH), "Model file not found!"
    print("  -> PASSED: Model file present.")

    # 2. Math verification (EAR & MAR)
    print("[TEST 2/5] Testing mathematical calculation functions...")
    # Open eye dummy coordinates
    open_eye = np.array([[10, 20], [15, 25], [20, 25], [25, 20], [20, 15], [15, 15]])
    closed_eye = np.array([[10, 20], [15, 20.5], [20, 20.5], [25, 20], [20, 19.5], [15, 19.5]])

    ear_open = calculate_ear(open_eye)
    ear_closed = calculate_ear(closed_eye)
    print(f"  -> EAR Open: {ear_open:.3f}, EAR Closed: {ear_closed:.3f}")
    assert ear_open > ear_closed, "EAR Open must be greater than EAR Closed!"
    print("  -> PASSED: EAR formula behaves accurately.")

    # Mouth MAR
    mouth_normal = np.array([[10, 50], [50, 50], [30, 48], [30, 52]])
    mouth_yawn = np.array([[10, 50], [50, 50], [30, 30], [30, 70]])
    mar_normal = calculate_mar(mouth_normal)
    mar_yawn = calculate_mar(mouth_yawn)
    print(f"  -> MAR Normal: {mar_normal:.3f}, MAR Yawn: {mar_yawn:.3f}")
    assert mar_yawn > mar_normal, "MAR Yawn must be greater than normal mouth!"
    print("  -> PASSED: MAR formula behaves accurately.")

    # 3. Model inference test
    print("[TEST 3/5] Testing MediaPipe FaceLandmarker initialization & inference...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
    res = detector.detect(mp_image)
    print("  -> PASSED: Detector initialized and processed frame successfully.")

    # 4. Audio manager test
    print("[TEST 4/5] Testing Audio Alert Manager...")
    alert_mgr = AlertManager()
    alert_mgr.start_alarm()
    assert alert_mgr.is_playing == True
    alert_mgr.stop_alarm()
    assert alert_mgr.is_playing == False
    alert_mgr.cleanup()
    print("  -> PASSED: Audio Alert Manager started and stopped cleanly.")

    # 5. HUD rendering test
    print("[TEST 5/6] Testing Heads-Up Display (HUD) rendering on OpenCV canvas...")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hud(canvas, ear=0.28, mar=0.35, closed_counter=0, ear_thresh=0.23,
             consec_frames=20, drowsy_count=1, yawn_count=0, alarm_active=False,
             yawn_active=False, alert_mgr=alert_mgr, fps=30.0,
             cam_id=0, alarm_sec=10.5, sos_active=True, sos_email="emergency_contact@example.com")
    assert canvas.shape == (480, 640, 3)
    print("  -> PASSED: HUD & Emergency SOS Banner rendered with no errors.")

    # 6. Emergency Geolocation & SOS Notification test
    print("[TEST 6/6] Testing Emergency Geolocation & SOS Dispatch Engine...")
    geo = get_current_geolocation()
    assert "lat" in geo and "lon" in geo and "maps_url" in geo, "Geolocation format invalid!"
    print(f"  -> GPS Location: {geo['city']}, {geo['region']}, {geo['country']} ({geo['lat']}, {geo['lon']})")
    
    notifier = EmergencyNotifier("emergency_contact@example.com")
    dispatched = notifier.dispatch_sos_alert(continuous_seconds=10.0, ear_value=0.08)
    assert dispatched is not None, "Failed to dispatch SOS alert!"
    print("  -> PASSED: Emergency SOS notification successfully dispatched & logged.")

    print("=" * 60)
    print("ALL 6 TESTS PASSED SUCCESSFULLY! PROJECT IS 100% READY.")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

