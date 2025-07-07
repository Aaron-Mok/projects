import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal
from collections import deque

# === Globals ===

fft_buffer = deque(maxlen=1)
baseline_spectrum = None
baseline_frames = []
baseline_duration_sec = 3  # Seconds of ambient noise to capture
baseline_done = False

# === Settings ===
samplerate = 44100
blocksize = 4096
window = np.hanning(blocksize)
min_freq = 90
max_freq = 4186  # C8

freqs = np.fft.rfftfreq(blocksize, 1 / samplerate)
latest_spectrum = np.zeros_like(freqs)
current_threshold = np.zeros_like(freqs)
detected_notes = []
note_buffer = {}

# === Frequency-dependent threshold profile ===
threshold_profile = 1 / (freqs + 1)  # Avoid division by zero
threshold_profile /= np.max(threshold_profile)  # Normalize to [0, 1]

# === Frequency to Note ===
def freq_to_note(freq):
    A4 = 440
    if freq <= 0:
        return None
    n = round(12 * np.log2(freq / A4)) + 69  # MIDI note number
    note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    note = note_names[n % 12]
    octave = n // 12 - 1
    return note + str(octave)

    # A0 = A4 / (2 ** 4)
    # semitone = round((np.log(freq / A4) / np.log(2)) * 12) + 9
    # octave = round(np.log(freq / A0) / np.log(2))
    # note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab','A', 'Bb', 'B']
    # return note_names[semitone % 12] + str(octave)

# === High-pass filter ===
def highpass_filter(signal, cutoff=90, fs=44100, order=3):
    b, a = scipy.signal.butter(order, cutoff / (fs / 2), btype='high')
    return scipy.signal.lfilter(b, a, signal)

# === Audio callback ===
def audio_callback(indata, frames, time, status):
    global latest_spectrum, detected_notes, note_buffer, current_threshold
    global baseline_spectrum, baseline_frames, baseline_done

    samples = highpass_filter(indata[:, 0], cutoff=90) * window
    spectrum = np.abs(np.fft.rfft(samples))
    spectrum = np.maximum(spectrum, 0)

    # === Baseline collection ===
    if not baseline_done:
        baseline_frames.append(spectrum)
        total_collected = len(baseline_frames) * blocksize / samplerate
        print(f"⏳ Calibrating baseline noise... {total_collected:.1f}s")
        latest_spectrum[:] = spectrum  # Display raw spectrum during calibration
        current_threshold[:] = np.zeros_like(freqs) + 4 # No threshold yet
        if total_collected >= baseline_duration_sec:
            baseline_spectrum = np.mean(baseline_frames, axis=0)
            baseline_done = True
            print("✅ Baseline noise profile captured.")
        return

    # Subtract baseline
    cleaned_spectrum = spectrum - baseline_spectrum
    cleaned_spectrum = np.clip(cleaned_spectrum, 0, None)

    # Average
    fft_buffer.append(cleaned_spectrum)
    avg_spectrum = np.mean(fft_buffer, axis=0)
    latest_spectrum = avg_spectrum

    # Threshold
    dynamic_scalar = np.mean(avg_spectrum) + 4 * np.std(avg_spectrum)
    current_threshold = dynamic_scalar * threshold_profile + 4

    # print("Current Threshold:", current_threshold)

    # valid = (avg_spectrum >= current_threshold) & (avg_spectrum >= 2)
    valid = (avg_spectrum >= current_threshold)
    valid_indices = np.where(valid)[0]

    # if len(valid_indices) > 0:
    #     max_index = valid_indices[np.argmax(avg_spectrum[valid_indices])]
    #     max_freq_detected = freqs[max_index]
    #     note = freq_to_note(max_freq_detected)
    #     detected_notes[:] = [note] if note else []
    #     print("🎵 Note:", detected_notes)
    # else:
    #     detected_notes[:] = []
    if len(valid_indices) > 0:
        detected_notes[:] = []
        for idx in valid_indices:
            freq = freqs[idx]
            note = freq_to_note(freq)
            if note and note not in detected_notes:
                detected_notes.append(note)
        print("🎵 Notes:", detected_notes)
    else:
        detected_notes[:] = []

# === Plot setup ===
plt.style.use('./style.mplstyle')
fig, ax = plt.subplots()
line, = ax.plot(freqs, latest_spectrum, label="Spectrum")
threshold_line, = ax.plot(freqs, np.full_like(freqs, current_threshold), 'r--', label="Threshold")
ax.legend()

ax.set_xlim(min_freq, max_freq)
ax.set_ylim(0, 10)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Real-Time FFT of audio from webcam mic")

def update_plot(frame):
    line.set_ydata(latest_spectrum)
    threshold_line.set_ydata(current_threshold)
    return line, threshold_line

ani = FuncAnimation(fig, update_plot, interval=50, blit=False)

# === Run ===
if __name__ == "__main__":
    print("🎙️ Using webcam mic (device=1). Close plot window to stop.")
    with sd.InputStream(device=1, channels=1, samplerate=samplerate,
                        blocksize=blocksize, callback=audio_callback):
        plt.show()
