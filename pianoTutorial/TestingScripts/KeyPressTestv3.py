import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal
from collections import deque
fft_buffer = deque(maxlen=1)  # average over last 4 FFT frames

# === Settings ===
samplerate = 44100
blocksize = 4096  # longer window for better resolution
window = np.hanning(blocksize)
min_freq = 90    # high-pass cutoff + ignore sub-audio
max_freq = 4186  # C8

freqs = np.fft.rfftfreq(blocksize, 1 / samplerate)
latest_spectrum = np.zeros_like(freqs)
detected_notes = []
note_buffer = {}

# === Frequency to Note ===
def freq_to_note(freq):
    A4 = 440
    if freq <= 0:
        return None
    n = round(12 * np.log2(freq / A4)) + 69
    note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    return note_names[n % 12] + str(n // 12 - 1)

# === High-pass filter ===
def highpass_filter(signal, cutoff=90, fs=44100, order=3):
    b, a = scipy.signal.butter(order, cutoff / (fs / 2), btype='high')
    return scipy.signal.lfilter(b, a, signal)

# === Audio callback ===
def audio_callback(indata, frames, time, status):
    global latest_spectrum, detected_notes, note_buffer

    # Apply high-pass filter
    samples = highpass_filter(indata[:, 0], cutoff=90) * window

    # FFT this frame
    spectrum = np.abs(np.fft.rfft(samples))
    spectrum = np.maximum(spectrum, 0)

    # Append and average
    fft_buffer.append(spectrum)
    avg_spectrum = np.mean(fft_buffer, axis=0)
    latest_spectrum = avg_spectrum  # for display

    # Threshold and peak detection on smoothed spectrum
    dynamic_thresh = np.mean(avg_spectrum) + 4 * np.std(avg_spectrum)
    peaks, _ = scipy.signal.find_peaks(avg_spectrum, height=2.5, distance=15)

    # Filter frequencies
    freqs_peaked = freqs[peaks]
    detected_freqs = []
    for f in freqs_peaked:
        if min_freq <= f <= max_freq:
            if not any(abs(f - hf*2) < 10 or abs(f - hf/2) < 10 for hf in detected_freqs):
                detected_freqs.append(f)

    # Convert to notes
    current_frame_notes = list(set(filter(None, [freq_to_note(f) for f in detected_freqs])))

    # Debounce (note must appear 2+ times)
    for note in current_frame_notes:
        note_buffer[note] = note_buffer.get(note, 0) + 1
    for note in list(note_buffer.keys()):
        if note not in current_frame_notes:
            note_buffer[note] -= 1
            if note_buffer[note] <= 0:
                del note_buffer[note]

    detected_notes[:] = [note for note, count in note_buffer.items() if count >= 2]

    if detected_notes:
        print("🎵 Notes:", detected_notes)

# === Plot setup ===
fig, ax = plt.subplots()
line, = ax.plot(freqs, latest_spectrum)
ax.set_xlim(min_freq, max_freq)
ax.set_ylim(0, 10)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Cleaned Real-Time FFT with Note Detection")

def update_plot(frame):
    line.set_ydata(latest_spectrum)
    return line,

ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

# === Run ===
if __name__ == "__main__":
    print("🎙️ Using webcam mic (device=1). Close plot window to stop.")
    with sd.InputStream(device=1, channels=1, samplerate=samplerate,
                        blocksize=blocksize, callback=audio_callback):
        plt.show()
