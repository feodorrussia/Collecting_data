import os

import numpy as np
from scipy import fft, ifft
import matplotlib.pyplot as plt


def showNoiseGist(vals):
    a_ = np.array([(v.real ** 2 + v.imag ** 2) ** 0.5 for v in vals])
    phase_ = np.array([-np.tanh(v.real / v.imag) if v.imag != 0 else np.pi for v in vals])

    plt.figure()
    plt.hist(a_, 30, density=True, label="A")
    plt.hist(phase_, 30, density=True, label="Phase")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


# Parameters
num_samples = 1000  # Number of samples in the noise signal
sampling_freq = 1000  # Sampling frequency in Hz

# Generate frequency axis
freq_axis = np.fft.fftfreq(num_samples, d=1 / sampling_freq)

# Create r Gaussian amplitude spectrum
mean_freq = 100  # Mean frequency in Hz
std_dev_freq = 10  # Standard deviation of frequencies in Hz
amplitude_spec = np.random.default_rng().normal(loc=mean_freq, scale=std_dev_freq, size=num_samples)

# Generate random phase spectrum
phase_spec = np.random.uniform(0, 2 * np.pi, num_samples)

# Combine amplitude and phase spectra to get the complex spectrum
complex_spec = amplitude_spec * np.exp(1j * phase_spec)

showNoiseGist(complex_spec)

# Perform inverse Fourier transform to get the noise signal
noise_signal = np.real(ifft(complex_spec))

# Plotting
plt.figure(figsize=(10, 4))
plt.plot(np.arange(num_samples) / sampling_freq, noise_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Noise Signal with Gaussian Amplitude of Frequencies')
plt.grid(True)
plt.show()

# save_noise
