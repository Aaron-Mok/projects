import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Parameters
fs = 10e8       # Sampling frequency (Hz)
duration = 3e-6  # Signal duration (seconds)

source_modulation = 10e6  # Frequency of the modulation (Hz)
source_amplitude = 1    # Amplitude of the desired signal

noise_std = 0.1   # Amplitude of the noise
reflectivity = 0.1      # Reflectivity of the object is 10%

distance = 1                # Distance between the light source and the object (m)
c = 3e8                     # Speed of light (m/s)
time_ABA = 2*distance/c
phase_signal = 2*np.pi*source_modulation*(time_ABA)    # Phase of the detected signal (radians)

# Time vector
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate the light source
signal = source_amplitude * np.sin(2 * np.pi * source_modulation * t)

# Generate the detected signal
np.random.seed(1)
noise = noise_std * np.random.normal(size=t.shape)
detected_signal = reflectivity * source_amplitude * np.sin(2 * np.pi * source_modulation * t + phase_signal) + noise

# Reference signals (sin and cos for lock-in detection)
f_reference = source_modulation
ref_sin = np.sin(2 * np.pi * f_reference * t)
ref_cos = np.cos(2 * np.pi * f_reference * t)

# Perform lock-in detection
I = detected_signal * ref_sin   # In-phase component
Q = detected_signal * ref_cos   # Quadrature component

# Fourier Transform of the noisy signal
freqs = np.fft.fftfreq(len(detected_signal), 1 / fs)
fft_detected_signal_mixed_sin = np.abs(fft(I)) / len(I)
fft_detected_signal_mixed_cos = np.abs(fft(Q)) / len(Q)

# DC component of the mixed signals
DC_I = np.mean(I)
DC_Q = np.mean(Q)

# Calculate amplitude and phase
amplitude_detected = 2 * np.sqrt(DC_I**2 + DC_Q**2)
phase_detected = np.arctan2(DC_Q, DC_I)

time_detected = phase_detected/(2*np.pi*source_modulation)
distance_detected = time_detected*c/2

# Display results
print(f"Detected Amplitude: {amplitude_detected:.3f}")
print(f"Detected Phase: {phase_detected:.3f} rad")
print(distance_detected)

# Plotting
plt.style.use('./style.mplstyle')
plt.subplots(4, 1, figsize=(10, 8))

# Amplitude modulated light source
plt.subplot(4, 1, 1)
plt.plot(t, signal, linewidth=1)
plt.title("Amplitude modulated light source")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Detected signal
plt.subplot(4, 1, 2)
plt.plot(t, detected_signal, label="Detected signal (noisy)", linewidth=1)
plt.plot(t, reflectivity*signal, label="Noiseless phase shifted light", linewidth=1)
plt.legend()
plt.title("Detected light signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Mixed signals
plt.subplot(4, 1, 3)
plt.plot(t, I, label= 'In-phase component' + r" $I(t)$")
plt.legend()
plt.title("Detected signal multiplied with" + r" $R_1(t)$")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Fourier Transform
plt.subplot(4, 1, 4)
plt.plot(freqs[:len(freqs)//2], fft_detected_signal_mixed_sin[:len(freqs)//2])
plt.title("Fourier Transform of In-phase component" + r" $I(t)$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 10*source_modulation)

plt.tight_layout()

plt.subplots(figsize=(10, 2))
plt.plot(t, signal, linewidth=1)
plt.title("Amplitude modulated light source")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplots(figsize=(10, 2))
plt.plot(t, detected_signal, label="detected signal", linewidth=1)
plt.plot(t, reflectivity*signal, label="noiseless phase-shifted light", linewidth=1)
plt.legend()
plt.title("Detected light signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplots(figsize=(10, 4))
plt.plot(t, I, label= 'In-phase component' + r" $I(t)$")
plt.legend()
plt.title("Detected signal multiplied with" + r" $R_1(t)$")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplots(figsize=(10, 2))
plt.plot(freqs[:len(freqs)//2], fft_detected_signal_mixed_sin[:len(freqs)//2])
plt.title("Fourier Transform of In-phase component" + r" $I(t)$")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 10*source_modulation)

plt.show()


