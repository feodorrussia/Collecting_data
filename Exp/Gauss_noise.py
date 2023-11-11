import os

import numpy as np
from scipy.stats import norm
from scipy import fft
import matplotlib.pyplot as plt


def show_Noise_RP_Gist(vals):
    a_ = np.array([(v.real ** 2 + v.imag ** 2) ** 0.5 for v in vals])
    phase_ = np.array([np.arctan(v.imag / v.real) if v.real != 0 else np.pi / 2 for v in vals])

    plt.figure()
    plt.hist(a_, 30, density=True, label="R")
    plt.hist(phase_, 30, density=True, label="Phase")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def show_Noise_ReIm_Gist(vals):

    plt.figure()
    plt.grid(True)
    plt.hist(vals.real, 30, density=True, label="Re")
    plt.hist(vals.imag, 30, density=True, label="Im")
    plt.legend(loc='best')
    plt.show()


def save_Noise_toFile(data, name_f="noise_Gauss_000Hz_Rng000_v0", path_noise="Datas/noises/"):
    if not os.path.isdir(path_noise):
        os.makedirs(path_noise)

    with open(path_noise + name_f + ".txt", "w") as file:
        file.writelines([f"{str(i)} {str(data[i])}\n" for i in range(len(data))])


# Gauss
# mu_a, sigma_a = 2, 3
# mu_phase, sigma_phase = 0, np.pi / 2
# r = np.random.default_rng().normal(loc=mu_a, scale=sigma_a, size=(1000,))
# phase = np.random.default_rng().normal(loc=mu_phase, scale=sigma_phase, size=(1000,))

mu_re, sigma_re = 1, 3
mu_im, sigma_im = mu_re, sigma_re  # 0, 1
real_part = np.random.default_rng().normal(loc=mu_re, scale=sigma_re, size=(1000,))
imag_part = np.random.default_rng().normal(loc=mu_im, scale=sigma_im, size=(1000,))

# plt.figure()
# plt.hist(r, 30, density=True, label="R")
# plt.hist(phase, 30, density=True, label="Phase")
# plt.legend(loc='best')
# plt.grid(True)
# plt.show()

# frequencies_values_gauss = r * np.cos(phase) + 1j * r * np.sin(phase)
frequencies_values_gauss = real_part + 1j * imag_part

noise_gauss = fft.ifft(frequencies_values_gauss)[1:]

# show_Noise_RP_Gist(frequencies_values_gauss)
show_Noise_ReIm_Gist(frequencies_values_gauss)

plt.figure()
plt.plot(np.linspace(0, noise_gauss.shape[-1], noise_gauss.shape[-1]), noise_gauss.real)
plt.grid(True)
plt.show()

version = 0
save_Noise_toFile(name_f=f"noise_Gauss_Mn{str(mu_re).rjust(3, '0')}_Std{str(sigma_re).rjust(3, '0')}_v{version}",
                  data=noise_gauss.real)
