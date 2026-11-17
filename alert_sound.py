"""
alert_sound.py - Non-blocking Audio Alert Manager for Drowsiness Detection System
Supports both Pygame audio playback (alarm.wav) and native Windows winsound.Beep fallback.
"""

import os
import sys
import math
import wave
import struct
import threading
import time

ALARM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alarm.wav")


def generate_default_alarm(filepath=ALARM_FILE):
    """Generates a pleasant yet urgent 2-tone pulsating buzzer WAV file."""
    if os.path.exists(filepath):
        return
    sample_rate = 44100
    duration = 1.0  # 1-second looping cycle
    freq1 = 2500    # High tone (Hz)
    freq2 = 1800    # Lower tone (Hz)

    with wave.open(filepath, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        num_samples = int(sample_rate * duration)
        data = bytearray()

        # 4 pulses per second
        for i in range(num_samples):
            t = i / sample_rate
            pulse_index = int(t * 4)
            sub_t = (t * 4) % 1.0

            # Beep for 75% of each pulse, silent for 25%
            if sub_t < 0.75:
                current_freq = freq1 if (pulse_index % 2 == 0) else freq2
                envelope = min(1.0, sub_t / 0.05, (0.75 - sub_t) / 0.05)
                val = int(32767.0 * 0.75 * envelope * math.sin(2.0 * math.pi * current_freq * t))
            else:
                val = 0

            data.extend(struct.pack('<h', val))
        wav.writeframes(data)


class AlertManager:
    """Thread-safe, non-blocking alarm manager."""

    def __init__(self, sound_file=ALARM_FILE):
        self.sound_file = sound_file
        self.is_playing = False
        self.muted = False
        self._lock = threading.Lock()
        self._pygame_loaded = False
        self._winsound_thread = None
        self._stop_winsound = threading.Event()

        # Ensure alarm file exists
        generate_default_alarm(self.sound_file)

        # Initialize Pygame mixer
        try:
            import pygame
            pygame.mixer.init()
            if os.path.exists(self.sound_file):
                self._sound = pygame.mixer.Sound(self.sound_file)
                self._pygame_loaded = True
        except Exception:
            self._pygame_loaded = False

    def start_alarm(self):
        """Starts alarm playback asynchronously without blocking video processing."""
        if self.muted:
            return

        with self._lock:
            if self.is_playing:
                return
            self.is_playing = True

        if self._pygame_loaded:
            try:
                self._sound.play(loops=-1)
                return
            except Exception:
                pass

        # Fallback to Windows native winsound in background thread
        if sys.platform == "win32":
            self._stop_winsound.clear()
            self._winsound_thread = threading.Thread(target=self._winsound_loop, daemon=True)
            self._winsound_thread.start()

    def _winsound_loop(self):
        """Native Windows Beep loop in daemon thread."""
        try:
            import winsound
            while not self._stop_winsound.is_set():
                winsound.Beep(2500, 250)
                time.sleep(0.05)
        except Exception:
            pass

    def stop_alarm(self):
        """Stops the alarm immediately."""
        with self._lock:
            if not self.is_playing:
                return
            self.is_playing = False

        if self._pygame_loaded:
            try:
                self._sound.stop()
            except Exception:
                pass

        if self._winsound_thread and self._winsound_thread.is_alive():
            self._stop_winsound.set()
            self._winsound_thread = None

    def toggle_mute(self):
        """Toggle mute state on or off."""
        self.muted = not self.muted
        if self.muted and self.is_playing:
            self.stop_alarm()
        return self.muted

    def cleanup(self):
        """Clean up audio resources on application exit."""
        self.stop_alarm()
        if self._pygame_loaded:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass


if __name__ == "__main__":
    print("Testing AlertManager...")
    mgr = AlertManager()
    print("Playing alarm for 2 seconds...")
    mgr.start_alarm()
    time.sleep(2.0)
    print("Stopping alarm...")
    mgr.stop_alarm()
    mgr.cleanup()
    print("Test finished successfully!")

