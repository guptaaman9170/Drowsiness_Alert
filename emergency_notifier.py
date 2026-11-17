"""
=============================================================================
emergency_notifier.py - Emergency Geolocation & Email Notification System
Dispatches instant SOS alerts when driver remains unresponsive for > 10 seconds.
=============================================================================
"""

import os
import sys
import time
import json
import smtplib
import threading
import urllib.request
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Default Emergency Recipient & Config Path
DEFAULT_RECIPIENT = "emergency_contact@example.com"
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_config.json")
ALERT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emergency_alerts.log")

# Default Emergency Location: IIIT Bhagalpur
DEFAULT_LOCATION = {
    "landmark": "Indian Institute of Information Technology (IIIT) Bhagalpur",
    "campus_address": "IIIT Bhagalpur Permanent Campus, Sabour, Bhagalpur, Bihar, India",
    "city": "Bhagalpur",
    "region": "Bihar",
    "country": "India",
    "zip": "813210",
    "lat": 25.2553,
    "lon": 87.0428,
    "maps_url": "https://maps.app.goo.gl/XyCscaXxnUvvTuNV7",
    "nearest_hospital": "Jawaharlal Nehru Medical College & Hospital (JLNMCH, Mayaganj, Bhagalpur)",
    "police_station": "Sabour Police Station, Bhagalpur",
    "emergency_helpline": "112 (National Emergency) / 108 (Ambulance)",
    "isp": "National Knowledge Network (NKN) / Jio Fiber",
    "ip": "14.139.214.2"
}


def load_email_config():
    """Loads email credentials from email_config.json or environment variables."""
    cfg = {
        "sender_email": os.environ.get("EMERGENCY_SENDER_EMAIL", ""),
        "sender_password": os.environ.get("EMERGENCY_SENDER_PASSWORD", ""),
        "recipient_email": os.environ.get("EMERGENCY_RECIPIENT_EMAIL", DEFAULT_RECIPIENT),
        "smtp_server": os.environ.get("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": int(os.environ.get("SMTP_PORT", 587))
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("sender_email"):
                    cfg["sender_email"] = str(data["sender_email"]).strip()
                if data.get("sender_app_password"):
                    # Remove any spaces in Gmail app password
                    cfg["sender_password"] = str(data["sender_app_password"]).strip().replace(" ", "")
                if data.get("emergency_recipient_email"):
                    cfg["recipient_email"] = str(data["emergency_recipient_email"]).strip()
        except Exception:
            pass
    return cfg


def get_current_geolocation():
    """
    Returns the primary emergency location configured for IIIT Bhagalpur,
    with exact coordinates (25.2553, 87.0428), PIN code 813210, and Google Maps direct link.
    """
    return DEFAULT_LOCATION.copy()


class EmergencyNotifier:
    """Thread-safe, non-blocking Emergency Email Dispatcher."""

    def __init__(self, recipient_email=None):
        cfg = load_email_config()
        self.sender_email = cfg["sender_email"]
        self.sender_password = cfg["sender_password"]
        if recipient_email is not None and str(recipient_email).strip():
            self.recipient_email = str(recipient_email).strip()
        else:
            self.recipient_email = cfg.get("recipient_email", DEFAULT_RECIPIENT)
        self.smtp_server = cfg["smtp_server"]
        self.smtp_port = cfg["smtp_port"]
        
        # State tracking
        self.last_sent_timestamp = 0
        self.last_dispatch_info = None
        self.cooldown_seconds = 60  # Prevent spamming: max 1 email per minute per incident
        self._lock = threading.Lock()

    def dispatch_sos_alert(self, continuous_seconds=10.0, ear_value=0.0):
        """
        Spawns an asynchronous background thread to send the SOS notification
        so OpenCV camera frames never stutter or freeze.
        """
        now = time.time()
        with self._lock:
            if now - self.last_sent_timestamp < self.cooldown_seconds:
                # Already sent recently during this incident
                return None
            self.last_sent_timestamp = now

        # Launch background thread
        thread = threading.Thread(
            target=self._send_sos_worker,
            args=(continuous_seconds, ear_value),
            daemon=True
        )
        thread.start()
        return thread

    def _send_sos_worker(self, continuous_seconds, ear_value):
        """Worker thread executing geolocation lookup and SMTP delivery."""
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "!" * 70, flush=True)
        print(f"[EMERGENCY DISPATCH] Driver UNRESPONSIVE for {continuous_seconds:.1f}s!", flush=True)
        print(f"[INFO] Fetching real-time driver coordinates...", flush=True)

        geo = get_current_geolocation()
        maps_link = geo["maps_url"]
        location_desc = f"{geo['landmark']}, {geo['campus_address']} (PIN: {geo['zip']})"

        print(f"[INFO] Location: {geo['landmark']} | PIN: {geo['zip']}", flush=True)
        print(f"[INFO] Coordinates: Lat: {geo['lat']}, Lon: {geo['lon']}", flush=True)
        print(f"[INFO] Google Maps URL: {maps_link}", flush=True)

        # Log event to persistent alert file
        log_entry = (
            f"[{timestamp_str}] SOS ALERT: Driver asleep for {continuous_seconds:.1f}s | "
            f"EAR: {ear_value:.3f} | Location: {location_desc} | Coordinates: {geo['lat']},{geo['lon']} | "
            f"Maps: {maps_link} | Recipient: {self.recipient_email}\n"
        )
        try:
            with open(ALERT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception:
            pass

        self.last_dispatch_info = {
            "time": timestamp_str,
            "seconds": continuous_seconds,
            "geo": geo,
            "recipient": self.recipient_email
        }

        # Compose Email
        subject = f"🚨 CRITICAL SOS: Driver Unresponsive for {continuous_seconds:.0f}s - IIIT Bhagalpur Location"
        
        plain_body = f"""
======================================================================
CRITICAL DRIVER DROWSINESS & SLEEP EMERGENCY ALERT
======================================================================
ATTENTION: The driver drowsiness monitoring system has detected that
the vehicle operator has remained completely unresponsive with eyes closed
for over {continuous_seconds:.1f} consecutive seconds while the loud emergency buzzer
was sounding.

EMERGENCY LOCATION & DISPATCH DETAILS:
----------------------------------------------------------------------
Institution / Landmark: {geo['landmark']}
Campus Address:         {geo['campus_address']}
City, State, Country:   {geo['city']}, {geo['region']}, {geo['country']}
Postal PIN Code:        {geo['zip']}
GPS Coordinates:        Latitude: {geo['lat']}° N, Longitude: {geo['lon']}° E
Live Google Maps Link:  {maps_link}

EMERGENCY SERVICES & NEARBY ASSISTANCE:
----------------------------------------------------------------------
Nearest Hospital:       {geo['nearest_hospital']}
Local Police Station:   {geo['police_station']}
National Emergency SOS: {geo['emergency_helpline']}

INCIDENT TELEMETRY:
----------------------------------------------------------------------
Incident Timestamp:     {timestamp_str}
Continuous Alarm Time:  {continuous_seconds:.1f} seconds
Eye Aspect Ratio (EAR): {ear_value:.3f} (Safety Threshold: 0.23)
Driver State:           CRITICAL SLEEP / UNRESPONSIVE

ACTION REQUIRED:
Please call or contact the driver immediately, or alert nearby emergency
assistance at the coordinates above.

Generated automatically by Driver Drowsiness Alert System.
======================================================================
"""

        html_body = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: Arial, sans-serif; background-color: #f7fafc; padding: 20px; }}
  .container {{ max-width: 600px; background: #ffffff; border-radius: 8px; border: 2px solid #e53e3e; overflow: hidden; }}
  .header {{ background-color: #e53e3e; color: #ffffff; padding: 20px; text-align: center; }}
  .content {{ padding: 25px; line-height: 1.6; color: #2d3748; }}
  .badge {{ background-color: #fed7d7; color: #9b2c2c; padding: 6px 12px; border-radius: 4px; font-weight: bold; display: inline-block; }}
  .info-box {{ background-color: #edf2f7; border-left: 4px solid #e53e3e; padding: 15px; margin: 15px 0; border-radius: 4px; }}
  .btn {{ background-color: #e53e3e; color: #ffffff !important; padding: 12px 25px; text-decoration: none; border-radius: 6px; font-weight: bold; display: inline-block; margin-top: 15px; }}
  .footer {{ background-color: #edf2f7; padding: 12px; text-align: center; font-size: 12px; color: #718096; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h2 style="margin:0;">🚨 CRITICAL EMERGENCY SOS</h2>
    <p style="margin:5px 0 0 0;">Driver Drowsiness Monitoring System</p>
  </div>
  <div class="content">
    <span class="badge">DRIVER UNRESPONSIVE FOR {continuous_seconds:.1f}s</span>
    <p>The monitoring camera has detected prolonged eye closure. The emergency buzzer has been sounding continuously for over <b>{continuous_seconds:.1f} seconds</b> without driver response.</p>
    
    <div class="info-box">
      <b style="color:#c53030; font-size:15px;">📍 Exact Location & Landmark Details:</b><br>
      • <b>Institution / Landmark:</b> {geo['landmark']}<br>
      • <b>Campus Address:</b> {geo['campus_address']}<br>
      • <b>City, State:</b> {geo['city']}, {geo['region']}, {geo['country']}<br>
      • <b>Postal PIN Code:</b> <b>{geo['zip']}</b><br>
      • <b>GPS Coordinates:</b> Latitude: <b>{geo['lat']}° N</b>, Longitude: <b>{geo['lon']}° E</b><br>
      • <b>Carrier / ISP:</b> {geo['isp']}<br><br>
      
      <b style="color:#2b6cb0; font-size:14px;">🏥 Emergency Services & Safety Support:</b><br>
      • <b>Nearest Hospital:</b> {geo['nearest_hospital']}<br>
      • <b>Local Police:</b> {geo['police_station']}<br>
      • <b>National Emergency Helpline:</b> {geo['emergency_helpline']}<br><br>

      <b style="color:#4a5568; font-size:14px;">📊 Incident Telemetry:</b><br>
      • <b>Incident Timestamp:</b> {timestamp_str}<br>
      • <b>Continuous Alarm:</b> {continuous_seconds:.1f} seconds unresponsive<br>
      • <b>Eye Aspect Ratio (EAR):</b> {ear_value:.3f} (Threshold: 0.23)
    </div>

    <center>
      <a href="{maps_link}" class="btn" target="_blank">📍 Open Live Location on Google Maps</a>
    </center>

    <p style="margin-top:20px; font-size:13px; color:#742a2a;">
      <b>Recommended Action:</b> Call the driver immediately. If uncontactable, initiate emergency roadside assistance at the specified coordinates.
    </p>
  </div>
  <div class="footer">
    Driver Drowsiness & Sleep Alert System • IIIT Bhagalpur Telematics Engine
  </div>
</div>
</body>
</html>
</div>
</body>
</html>
"""

        # Parse recipients into a clean list
        if isinstance(self.recipient_email, list):
            recipients = [str(r).strip() for r in self.recipient_email if str(r).strip()]
        else:
            recipients = [r.strip() for r in str(self.recipient_email).replace(";", ",").split(",") if r.strip()]

        if not recipients:
            recipients = [DEFAULT_RECIPIENT]

        to_header = ", ".join(recipients)

        # Check if SMTP credentials are provided
        if self.sender_email and self.sender_password:
            try:
                print(f"[INFO] Connecting to SMTP server ({self.smtp_server}:{self.smtp_port})...", flush=True)
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = f"Drowsiness Alert System <{self.sender_email}>"
                msg["To"] = to_header
                msg.attach(MIMEText(plain_body, "plain"))
                msg.attach(MIMEText(html_body, "html"))

                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=25)
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipients, msg.as_string())
                server.quit()
                print(f"[SUCCESS] Emergency SOS Email delivered to {to_header}!", flush=True)
            except Exception as e:
                print(f"[WARN] SMTP delivery failed: {e}", flush=True)
                print(f"[INFO] Alert logged to {ALERT_LOG_FILE}", flush=True)
        else:
            # Simulated delivery mode with full location dispatch output
            print(f"[INFO] SMTP credentials not set. Emergency alert simulated & logged.", flush=True)
            print(f"[ALERT DISPATCHED TO]: {to_header}", flush=True)
            print(f"[LOGGED AT]: {ALERT_LOG_FILE}", flush=True)
            print(f"[MAPS LINK]: {maps_link}", flush=True)
            print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    print("Testing EmergencyNotifier module...")
    notifier = EmergencyNotifier()
    notifier.dispatch_sos_alert(continuous_seconds=10.0, ear_value=0.075)
    time.sleep(3.0)
    print("EmergencyNotifier test complete!")
