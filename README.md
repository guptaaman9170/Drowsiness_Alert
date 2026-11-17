# Driver Drowsiness & Sleep Alert System 🚗💤

An end-to-end Computer Vision project that continuously monitors a driver's eyes and face in real-time. When signs of fatigue, prolonged eye closure (sleeping), or yawning are detected, it immediately sounds a loud beep alarm to wake the person up and prevent accidents.

> **Assignment Submission Ready**: Built based on the classic **[GeeksforGeeks Drowsiness Detection Pipeline](https://www.geeksforgeeks.org/python/python-opencv-drowsiness-detection/)** using the **Eye Aspect Ratio (EAR)** algorithm by Soukupová & Čech (2016).

---

## 🌟 Key Features

- **Real-Time Eye Aspect Ratio (EAR) Computation**: Measures eye openness on every frame with sub-pixel precision.
- **Audible Beep Alarm**: Sounds an immediate loud alert buzzer whenever eyes remain closed beyond the safety threshold (default: ~0.8 seconds / 22 frames).
- **Non-Blocking Multi-Threaded Audio**: Sound runs in an asynchronous background thread so the video frame rate never drops or stutters during alarms.
- **Yawn Detection (Bonus MAR)**: Calculates Mouth Aspect Ratio (MAR) to detect early-stage driver drowsiness before their eyes even close.
- **Real-Time HUD (Heads-Up Display)**: Shows EAR progress bars, thresholds, FPS, event counters, and eye contour polygons directly on screen.
- **Interactive Controls**: Dynamically increase/decrease the EAR threshold with `+`/`-` keys, mute audio with `M`, or reset counters with `R`.
- **Three Implementations Included**:
  1. `main.py`: Production-grade implementation using Google MediaPipe Face Mesh & OpenCV. (Fastest, ~600 FPS inference, auto-downloads model).
  2. `gfg_dlib_drowsiness.py`: The exact classic GeeksforGeeks implementation using Dlib's 68-landmark predictor.
  3. `opencv_haar_drowsiness.py`: Pure OpenCV Haar Cascade implementation (zero external models needed).
- **10-Second Continuous Alarm Emergency SOS**: If the driver remains asleep/unresponsive with the alarm sounding for over 10 consecutive seconds, the system automatically dispatches an emergency SOS email notification with real-time geolocation, GPS coordinates, nearest hospital/police station, and a direct Google Maps link.

---

## 📐 How It Works: The Eye Aspect Ratio (EAR) Algorithm

For each eye, 6 landmarks $(p_1, p_2, p_3, p_4, p_5, p_6)$ are tracked:

```
        p2     p3
     .------.------.
  p1                p4   <--- (Horizontal Eye Landmarks)
     .------.------.
        p6     p5
```

The **Eye Aspect Ratio (EAR)** is computed as:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$

- **Eyes Open (Awake)**: Vertical distance is large $\implies \text{EAR} \approx 0.28 - 0.38$.
- **Blink**: Vertical distance drops rapidly for $3 - 8$ frames ($\approx 100 - 250\text{ ms}$) and recovers.
- **Sleeping / Drowsy**: Vertical distance drops to $\approx 0.05 - 0.15$ and stays low for $> 20$ consecutive frames.
- **Trigger**: Once $\text{EAR} < \text{Threshold}$ for $\ge 22$ consecutive frames, the alarm beeps continuously until eyes open.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Open PowerShell or Command Prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Main System
```bash
python main.py
```
*(Or simply double click `run.bat` on Windows!)*

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|:---:|:---|
| **`q`** or **`ESC`** | Exit the application cleanly |
| **`c`** | **Switch to next connected camera live** (instant toggle) |
| **`m`** | Toggle Mute / Unmute the beep alarm |
| **`+`** / **`=`** | Increase EAR sensitivity threshold (triggers alarm faster) |
| **`-`** / **`_`** | Decrease EAR sensitivity threshold |
| **`r`** | Reset session statistics (alerts, yawns, frames) |

---

## 📷 Camera Selection & Permissions

When you launch `main.py`, it automatically scans all connected camera devices and asks you:

```text
=================================================================
📷 CAMERA PERMISSION & SELECTION MENU
=================================================================
Found 2 connected video capture device(s):
   [0] -> Camera Index 0 (Resolution: 640x480) (Primary / Integrated)
   [1] -> Camera Index 1 (Resolution: 1920x1080) (External / USB)
-----------------------------------------------------------------
👉 Press [ENTER] to use Camera [0]
👉 Or type a camera number (e.g. 0, 1, 2...)
👉 Or paste an IP Webcam stream URL (e.g. http://192.168.1.5:8080/video)
-----------------------------------------------------------------
Which camera do you want to use? [0]: 
```

You can also pass your preferred camera directly via command line:
```bash
python main.py -c 1
```

---

## 📁 Project Structure

```
sona_mana_hai/
│
├── main.py                     # Primary real-time drowsiness detector (MediaPipe + OpenCV + Audio + SOS)
├── alert_sound.py              # Threaded non-blocking audio alert manager (Pygame + Winsound)
├── emergency_notifier.py       # Asynchronous Emergency Geolocation & SOS Email Notifier
├── test_email.py               # Fast SMTP verification tool
├── email_config.example.json   # Template for email credentials configuration
├── generate_alarm.py           # Custom frequency wave audio generator
├── gfg_dlib_drowsiness.py      # Classic GeeksforGeeks Dlib 68-point reference code
├── opencv_haar_drowsiness.py   # Pure OpenCV Haar Cascades fallback detector
├── test_pipeline.py            # Automated test suite (all 6 integration tests)
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows 1-click batch launcher
├── alarm.wav                   # Generated 2-tone alert buzzer audio file
├── face_landmarker.task        # MediaPipe float16 face landmarker model
├── README.md                   # Project documentation
└── PROJECT_REPORT.md           # Full academic report with diagrams & viva questions
```

---

## 🚨 Emergency SOS Email Setup

1. Copy `email_config.example.json` to `email_config.json`:
   ```bash
   cp email_config.example.json email_config.json
   ```
2. Enter your Gmail address and 16-character [Gmail App Password](https://myaccount.google.com/apppasswords).
3. Test your configuration:
   ```bash
   python test_email.py
   ```
*(Note: `email_config.json` is protected by `.gitignore` and is never committed or pushed to GitHub).*

---

## 🧪 Testing the Project

You can run the automated validation tests anytime:
```bash
python test_pipeline.py
```
This tests:
1. Model loading integrity
2. EAR and MAR mathematical calculations
3. MediaPipe FaceLandmarker inference
4. Audio player startup and shutdown
5. OpenCV HUD drawing canvas

---

## 📚 Reference
- GeeksforGeeks: [Python OpenCV - Drowsiness Detection](https://www.geeksforgeeks.org/python/python-opencv-drowsiness-detection/)
- Tereza Soukupová and Jan Čech: *Real-Time Eye Blink Detection using Facial Landmarks*, Computer Vision Winter Workshop (CVWW), 2016.

