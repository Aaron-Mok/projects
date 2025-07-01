# audio/mic_listener.py

import numpy as np
import sounddevice as sd
import scipy.signal
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MicListener:
    def __init__(self, callback=None, samplerate=44100, blocksize=4096, baseline_duration_sec=3):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.window = np.hanning(blocksize)
        self.freqs = np.fft.rfftfreq(blocksize, 1 / samplerate)
        self.fft_buffer = deque(maxlen=1)

        self.baseline_duration_sec = baseline_duration_sec
        self.baseline_frames = []
        self.baseline_spectrum = None
        self.baseline_done = False

        self.latest_spectrum = np.zeros_like(self.freqs)
        self.current_threshold = np.zeros_like(self.freqs)
        self.detected_notes = []

        self.threshold_profile = 1 / (self.freqs + 1)
        self.threshold_profile /= np.max(self.threshold_profile)

        self.running = True
        self.callback = callback  # external callback when notes are detected

    def freq_to_note(self, freq):
        A4 = 440
        if freq <= 0:
            return None
        n = round(12 * np.log2(freq / A4)) + 69
        note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        return note_names[n % 12] + str(n // 12 - 1)

    def highpass_filter(self, signal, cutoff=90):
        b, a = scipy.signal.butter(3, cutoff / (self.samplerate / 2), btype='high')
        return scipy.signal.lfilter(b, a, signal)

    def audio_callback(self, indata, frames, time, status):
        samples = self.highpass_filter(indata[:, 0]) * self.window
        spectrum = np.abs(np.fft.rfft(samples))
        spectrum = np.maximum(spectrum, 0)

        if not self.baseline_done:
            self.baseline_frames.append(spectrum)
            collected = len(self.baseline_frames) * self.blocksize / self.samplerate
            print(f"⏳ Calibrating... {collected:.1f}s")
            self.latest_spectrum[:] = spectrum
            self.current_threshold[:] = 0
            if collected >= self.baseline_duration_sec:
                self.baseline_spectrum = np.mean(self.baseline_frames, axis=0)
                self.baseline_done = True
                print("✅ Baseline captured.")
            return

        cleaned = spectrum - self.baseline_spectrum
        cleaned = np.clip(cleaned, 0, None)

        self.fft_buffer.append(cleaned)
        avg_spectrum = np.mean(self.fft_buffer, axis=0)
        self.latest_spectrum = avg_spectrum

        scalar = np.mean(avg_spectrum) + 4 * np.std(avg_spectrum)
        self.current_threshold = scalar * self.threshold_profile

        valid = (avg_spectrum >= self.current_threshold) & (avg_spectrum >= 2)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) > 0:
            detected_notes = set()
            for idx in valid_indices:
                freq = self.freqs[idx]
                note = self.freq_to_note(freq)
                if note:
                    detected_notes.add(note)
            self.detected_notes = list(detected_notes)
            print("🎵 Note:", self.detected_notes)
            if self.callback:
                self.callback(self.detected_notes)
        else:
            self.detected_notes = []

    def run(self, device=1):
        with sd.InputStream(device=device, channels=1, samplerate=self.samplerate,
                            blocksize=self.blocksize, callback=self.audio_callback):
            print("🎙️ Mic listener started. Press Ctrl+C to stop.")
            import time
            while self.running:
                time.sleep(0.1)

    def stop(self):
        self.running = False