import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal
from collections import deque

current_threshold = 0
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
    global latest_spectrum, detected_notes, note_buffer, min_freq, max_freq, current_threshold

    # Apply high-pass filter
    samples = highpass_filter(indata[:, 0], cutoff=90) * window

    # FFT this frame
    spectrum = np.abs(np.fft.rfft(samples))
    spectrum = np.maximum(spectrum, 0)

    # Append and average
    fft_buffer.append(spectrum)
    avg_spectrum = np.mean(fft_buffer, axis=0)
    latest_spectrum = avg_spectrum  # for display

    # Find the strongest peak above amplitude threshold
    # Dynamic threshold (adaptive to noise floor)
    # dynamic_thresh = np.mean(avg_spectrum) + 4 * np.std(avg_spectrum)
    dynamic_thresh = 5 * np.mean(avg_spectrum)
    current_threshold = dynamic_thresh

    # Get all indices above dynamic threshold and within desired freq range
    # valid = (avg_spectrum >= dynamic_thresh) & (avg_spectrum>=4)
    valid = (avg_spectrum >= dynamic_thresh)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) > 0:
        # Find the index of the strongest peak among valid ones
        max_index = valid_indices[np.argmax(avg_spectrum[valid_indices])]
        max_freq = freqs[max_index]
        note = freq_to_note(max_freq)
        detected_notes[:] = [note] if note else []
        print("🎵 Note:", detected_notes)
    else:
        detected_notes[:] = []

# === Plot setup ===
fig, ax = plt.subplots()
line, = ax.plot(freqs, latest_spectrum, label="Spectrum")
threshold_line, = ax.plot(freqs, np.full_like(freqs, current_threshold), 'r--', label="Threshold")
ax.legend()

ax.set_xlim(min_freq, max_freq)
ax.set_ylim(0, 10)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Cleaned Real-Time FFT with Note Detection")

def update_plot(frame):
    line.set_ydata(latest_spectrum)
    threshold_line.set_ydata(np.full_like(freqs, current_threshold))
    return line, threshold_line

ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

# === Run ===
if __name__ == "__main__":
    print("🎙️ Using webcam mic (device=1). Close plot window to stop.")
    with sd.InputStream(device=1, channels=1, samplerate=samplerate,
                        blocksize=blocksize, callback=audio_callback):
        plt.show()
