import os
import random
import time

import numpy as np
from scipy import fft
import pandas as pd
import matplotlib.pyplot as plt

# Import library to clean output
import seaborn as sns


# Bresenham Line
def bresenham_line(matrix, i1, j1, i2, j2):
    result = []
    dx = abs(j2 - j1)
    dy = abs(i2 - i1)
    sx = 1 if j1 < j2 else -1
    sy = 1 if i1 < i2 else -1
    err = dx - dy
    while True:
        result.append(matrix[i1][j1])
        if j1 == j2 and i1 == i2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            j1 += sx
        if e2 < dx:
            err += dx
            i1 += sy
    return result


def add_Noise(data: np.ndarray, d=1e-1, name_noise_f="noise_Gauss_000Hz_Rng000_v0",
              path_noise="D:/Edu/Lab/Datas/noises/") -> np.ndarray:
    # with open(path_noise + name_noise_f + ".txt") as file:
    #     noise_data = np.array([float(line.split()[1]) for line in file.read().strip("\n").split("\n")])

    mu_re, sigma_re = 2, 3
    mu_im, sigma_im = mu_re, sigma_re  # 0, 1
    real_part = np.random.default_rng().normal(loc=mu_re, scale=sigma_re, size=(500,))
    imag_part = np.random.default_rng().normal(loc=mu_im, scale=sigma_im, size=(500,))

    frequencies_values_gauss = real_part + 1j * imag_part

    noise_data = fft.ifft(frequencies_values_gauss)[1:]

    start_ind = random.randint(0, noise_data.shape[-1] - data.shape[-1] - 1)
    noise_data = noise_data[start_ind:start_ind + data.shape[-1]]

    scale_noise = d * (max(data) - min(data)) / (max(noise_data) - min(noise_data))

    new_data = data + noise_data * scale_noise
    # show_Signal(new_data)

    return new_data


def show_Signal(vals):
    plt.figure(1)
    plt.plot(np.linspace(0, vals.shape[-1], vals.shape[-1]), vals)
    plt.grid()
    plt.show(block=False)
    plt.pause(4 / 5)
    plt.close(1)


def save_Filament_toFile(values: np.ndarray, name_f="29Hz/lines_v0/filament_map_0_000Hz_0000",
                         path_filaments="D:/Edu/Lab/Datas/filaments/maps/", add_data=None, auto_save=False):
    if add_data is None:
        add_data = [29, 0, [0, 0], [0, 0]]
    if not os.path.isdir(path_filaments):
        os.makedirs(path_filaments)

    # print(f"{name_f[:4]}_{name_f[11:14]}_{name_f[-3:]}.")

    if auto_save:
        zero_level = values[0]
        min_y, max_y = min(values), max(values)
        n_data = list(filter(lambda el: el != min_y and el != max_y, values))
        if len(n_data) == 0:
            show_Signal(values)
            save_fl = input(f"{name_f[:4]}_{name_f[11:14]}_{name_f[-3:]}. To don't save this signal, press [n]: ")
        else:
            min2_y, max2_y = min(n_data), max(n_data)
            if (abs((min_y - max_y)/zero_level) < 1e-1 or abs((min_y - zero_level)/(min2_y - zero_level)) > 2
                    or abs((max_y - zero_level)/(max2_y - zero_level)) > 2):
                print(abs((min_y - max_y)/zero_level), "< 0.1", abs((min_y - zero_level)/(min2_y - zero_level)), "> 2",
                      abs((max_y - zero_level)/(max2_y - zero_level)), "> 2")
                show_Signal(values)
                save_fl = input(f"{name_f[:4]}_{name_f[11:14]}_{name_f[-3:]}. To don't save this signal, press [n]: ")
            else:
                save_fl = "y"
    else:
        show_Signal(values)
        save_fl = input(f"{name_f[:4]}_{name_f[11:14]}_{name_f[-3:]}. To don't save this signal, press [n]: ")

    if not (add_data is None or save_fl.strip().lower() in ["n", "т"]):
        for d_i in range(100):
            scale_coef = 2 * 1e-1 + (random.random() - 0.5) * 3 * 1e-1
            values = add_Noise(values, d=scale_coef)

            data_add = [[add_data[0], add_data[3][0], add_data[2][0], add_data[3][1], add_data[2][1], ' '.join(values.real)]]
            dataframe_add = pd.DataFrame(data_add, columns=['Frequency', 'X_start', 'Y_start', 'X_end', 'Y_end', 'Values'])

            dataframe_add.to_csv(path_filaments + name_f, mode='a', header=False)
        return True
    return False


# Showing map with Bresenham Line plot
def show_Map_with_Line(c_map, line):
    sns.set()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    sns.heatmap(ax=ax[0], data=c_map)
    ax[0].plot(line[1], line[0], color="white", linewidth=2)
    sns.lineplot(ax=ax[1], data=bresenham_line(c_map, line[0][0], line[1][0], line[0][1], line[1][1]))
    plt.show()


# Set the number of files
n = 6000

# Set path to dir with maps files
path_to_map = "D:/Edu/Lab/Datas/maps/map_2/"
map_name = "map_2"
# Set scale coefficients of map
dx_map, dy_map = 4 * 1e-3, 4 * 1e-3

# Set path to dir for save filaments
path_to_filaments = "D:/Edu/Lab/Datas/filaments/maps/"

name_database = "total_database.csv"

# Making an array of positions with id
with open(path_to_map + "positions.txt", "r") as file_of_positions:
    lines = file_of_positions.readlines()[2:]
    lines = [line.replace("\n", "").split(" ") for line in lines]
    lines = [[round(float(line[0])), float(line[1]), float(line[2])] for line in lines]
    positions = pd.DataFrame(lines, columns=["id", "r", "z"])

# Create list with ids
ids = []
for i in range(n):
    name = path_to_map + "globus_" + str(i).rjust(4, "0") + ".txt"
    with open(name) as file_of_values:
        lines = file_of_values.readlines()[2:]
        lines = [line.split(" ")[:-1] for line in lines]
        lines = [[float(line[0]), float(line[1])] for line in lines]
    ids.append(lines)

# Making plot of the positions
x = sorted(list(set(positions.r)))
y = sorted(list(set(positions.z)))

# Creating r variable for the matrix
width = len(x)
height = len(y)
curr_map = np.zeros((height, width))

x_, y_ = np.meshgrid(x, y)

data = []
df = pd.DataFrame(data, columns=['Frequency', 'X_start', 'Y_start', 'X_end', 'Y_end', 'Values'])
df.to_csv(path_to_filaments + name_database)

for frequency in range(35, 60):
    order_num = 0
    for r, z in zip(np.hstack(x_), np.hstack(y_)):
        # print(r, z, sep="\n")
        id_pos = positions.loc[(positions["r"] == r) & (positions["z"] == z), 'id'].iloc[0]
        for ids_line in ids[id_pos]:
            if abs(ids_line[0] - frequency * 10 ** 9) < 10:
                curr_map[order_num // width][order_num % width] = ids_line[1]
        order_num += 1

    curr_map = curr_map - (np.abs(curr_map) > 1) * curr_map
    curr_map = np.round(curr_map, 4)

    # Make r line
    x_coord_top = int(28 + 0.19 * (65 - frequency))
    # coord_dx = 0
    for coord_dx in range(-5, 6):
        board_f = False
        version = 5 + coord_dx
        line_coord = [[0, 99], [x_coord_top, x_coord_top + coord_dx]]

        delta = 3
        h = 1

        k = width - max(line_coord[1]) - delta * h - 1  # 1  #
        for i in range(k):
            line_coord[1][1] += h
            line_coord[1][0] += h
            # show_Map_with_Line(curr_map, line_coord)

            board_f = save_Filament_toFile(
                name_f=name_database,
                values=np.array(
                    bresenham_line(curr_map, line_coord[0][0], line_coord[1][0], line_coord[0][1], line_coord[1][1])),
                add_data=[frequency, i, [line_coord[0][0] * dy_map, line_coord[0][1] * dy_map],
                          [line_coord[1][0] * dx_map, line_coord[1][1] * dx_map]], auto_save=board_f)
# 32Hz_v0/_000 46Hz_v1/_005 58Hz_v4/_000
#
