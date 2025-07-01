import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.signal
from scipy.signal import butter, lfilter

# === Settings ===
sd.default.device = (9, None)  # (input, output)
samplerate = 44100
duration = 0.1  # seconds per block
blocksize = int(samplerate * duration)
window = np.hanning(blocksize)
min_freq = 27.5   # A0
max_freq = 4186   # C8
freqs = np.fft.rfftfreq(blocksize, 1 / samplerate)

# === Globals ===
latest_spectrum = np.zeros_like(freqs)
noise_spectrum = np.zeros_like(freqs)
detected_notes = []
note_buffer = {}

# === Utility: Frequency to Note ===
def freq_to_note(freq):
    A4 = 440
    if freq <= 0: return None
    n = round(12 * np.log2(freq / A4)) + 69
    note_names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    return note_names[n % 12] + str(n // 12 - 1)

# === Step 1: Record Background Noise ===


def record_background_noise(duration_sec=2):
    print(sd.query_devices())
    print("🔇 Recording background noise... Stay quiet.")
    recording = sd.rec(int(samplerate * duration_sec), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Done recording background.")
    return recording[:, 0]

def compute_noise_spectrum(audio, blocksize):
    audio = audio[:len(audio) // blocksize * blocksize]
    num_blocks = len(audio) // blocksize
    total_spectrum = np.zeros(blocksize // 2 + 1)
    for i in range(num_blocks):
        chunk = audio[i*blocksize:(i+1)*blocksize] * window
        spectrum = np.abs(np.fft.rfft(chunk))
        total_spectrum += spectrum
    return total_spectrum / num_blocks

# === Step 2: Real-time Audio Callback with Note Detection ===
def audio_callback(indata, frames, time, status):
    global latest_spectrum, detected_notes, note_buffer
    samples = indata[:, 0] * window
    spectrum = np.abs(np.fft.rfft(samples))
    spectrum -= noise_spectrum
    spectrum = np.maximum(spectrum, 0)
    spectrum /= np.max(spectrum + 1e-8)
    latest_spectrum = spectrum

    # Dynamic thresholding
    dynamic_thresh = np.mean(spectrum) + 3 * np.std(spectrum)
    peaks, props = scipy.signal.find_peaks(spectrum, height=dynamic_thresh, distance=15)
    freqs_peaked = freqs[peaks]
    detected_freqs = []

    # Filter harmonics
    for f in freqs_peaked:
        if min_freq <= f <= max_freq:
            if not any(abs(f - hf*2) < 10 or abs(f - hf/2) < 10 for hf in detected_freqs):
                detected_freqs.append(f)

    # Convert to note names
    current_frame_notes = [freq_to_note(f) for f in detected_freqs]
    current_frame_notes = list(set(filter(None, current_frame_notes)))

    # Debounce: add to buffer
    for note in current_frame_notes:
        note_buffer[note] = note_buffer.get(note, 0) + 1
    for note in list(note_buffer.keys()):
        if note not in current_frame_notes:
            note_buffer[note] -= 1
            if note_buffer[note] <= 0:
                del note_buffer[note]

    # Only show sustained notes
    detected_notes[:] = [note for note, count in note_buffer.items() if count >= 2]

    if detected_notes:
        print("🎵 Notes:", detected_notes)

# === Step 3: Plot Setup ===
fig, ax = plt.subplots()
line, = ax.plot(freqs, latest_spectrum)
ax.set_xlim(min_freq, max_freq)
ax.set_ylim(0, 1)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Real-time FFT with Note Detection")

def update_plot(frame):
    line.set_ydata(latest_spectrum)
    return line,

ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

# === Main Execution ===
if __name__ == "__main__":
    background_audio = record_background_noise(duration_sec=2)
    noise_spectrum = compute_noise_spectrum(background_audio, blocksize)

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate, blocksize=blocksize):
        print("🎙️ Listening... Close the plot window to stop.")
        plt.show()
