import numpy as np
import scipy.fft as scf
import matplotlib.pyplot as plt

N = 400
T1 = 1 / 800
T2 = 2 / 500
x1 = np.linspace(0.0, N * T1, N, endpoint=False)
x2 = np.linspace(0.0, N * T1, N//2, endpoint=False)
vals1 = np.exp(50.0 * 1.j * 2.0 * np.pi * x1) + 0.5 * np.exp(-80.0 * 1.j * 2.0 * np.pi * x1)
vals2 = np.exp(50.0 * 1.j * 2.0 * np.pi * x2) + 0.5 * np.exp(-80.0 * 1.j * 2.0 * np.pi * x2)

yf1 = scf.fftshift(scf.fft(vals1))
xf1 = scf.fftshift(scf.fftfreq(N, T1))
yf2 = scf.fftshift(scf.fft(vals2))
xf2 = scf.fftshift(scf.fftfreq(N//2, T1))

# plt.plot(x1, vals1)
# plt.plot(x2, vals2)
# plt.plot(np.linspace(0.0, N * T1, 2*N, endpoint=False)[:-2], scf.irfft(yf1))
# plt.plot(np.linspace(0.0, N * T1, 2*N//2, endpoint=False)[:-2], scf.irfft(yf2))
plt.plot(xf1, 1.0 / N * np.abs(yf1))
plt.plot(xf2/2, 1.0 / N * np.abs(yf2))
plt.grid()
plt.show()
