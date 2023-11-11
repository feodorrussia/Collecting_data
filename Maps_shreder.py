import time
import matplotlib.pyplot as plot
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np


def show_Map(colormaps, data):
    n = len(colormaps)
    fig, axs = plot.subplots(1, n, figsize=(n * 2 + 2, 3), constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plot.show()


path = "Datas/maps/map_2/globus_"
width, height = [60, 100]  # input("input size (width height)").split()
curr_map = list()

with open(path + str(0).ljust(4, "0") + ".txt") as file:
    frequencies = [x[:4] for x in file.read().split()[7::3]]
    # print(f"Allow frequencies: {'; '.join(frequencies)}")
    index_freq = 36  # frequencies.index(input("Input frequency: "))
    # print(index_freq)
    I_Q_flag = 0  # int(input("Input 0 for I & 1 for Q"))

# for i in range(height):
#     row = []
#     for j in range(width):
#         # print(i * j)
#         with open(path + str(i * height + j).rjust(4, "0") + ".txt") as file:
#             value = float(file.read().split()[8 + I_Q_flag::3][index_freq])
#             row.append(value)
#     curr_map.append(row)


for i in range(6000):
    with open(path + str(i).rjust(4, "0") + ".txt") as file:
            value = float(file.read().split()[8 + I_Q_flag::3][index_freq])
            curr_map.append(value)

bottom = matplotlib.colormaps['Oranges_r'].resampled(128)
top = matplotlib.colormaps['Blues'].resampled(128)
viridis = matplotlib.colormaps['viridis'].resampled(8)

colors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
cmp = ListedColormap(colors, name='OrangeBlue')
np_map = np.matrix(curr_map).reshape((60, 100))
# np.random.seed(19680801)
# np_map = np.random.randn(30, 30)

print(np_map.shape)

show_Map([cmp], np_map)
