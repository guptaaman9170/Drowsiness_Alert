"""
generate_alarm.py - Custom Alarm Wave Generator
Creates a custom alarm.wav file with specified pitch and pulse rate.
"""

import math
import struct
import wave
import sys

def create_alarm_file(filename="alarm.wav", duration=1.0, high_freq=2500, low_freq=1800, sample_rate=44100):
    print(f"[INFO] Generating '{filename}' ({duration}s, {high_freq}Hz/{low_freq}Hz)...")
    with wave.open(filename, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        num_samples = int(sample_rate * duration)
        data = bytearray()

        for i in range(num_samples):
            t = i / sample_rate
            pulse_index = int(t * 4)  # 4 beeps per second
            sub_t = (t * 4) % 1.0

            if sub_t < 0.75:
                current_freq = high_freq if (pulse_index % 2 == 0) else low_freq
                envelope = min(1.0, sub_t / 0.05, (0.75 - sub_t) / 0.05)
                val = int(32767.0 * 0.75 * envelope * math.sin(2.0 * math.pi * current_freq * t))
            else:
                val = 0

            data.extend(struct.pack('<h', val))
        wav.writeframes(data)
    print(f"[INFO] '{filename}' successfully created!")

if __name__ == "__main__":
    create_alarm_file()

