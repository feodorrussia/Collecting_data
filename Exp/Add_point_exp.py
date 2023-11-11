import numpy as np
import pandas as pd
import scipy.interpolate as sc_i
import matplotlib.pyplot as plt


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


def increasing_Rate_cubic(values: np.array, rate, new_rate):
    f = sc_i.interp1d(np.linspace(0, len(values), len(values)), values, kind="cubic")
    return f(np.linspace(0, len(values), int(new_rate / rate * len(values))))


def increasing_Rate_quadratic(values: np.array, rate, new_rate):
    f = sc_i.interp1d(np.linspace(0, len(values), len(values)), values, kind="quadratic")
    return f(np.linspace(0, len(values), int(new_rate / rate * len(values))))


if __name__ == "__main__":
    # Set the number of files
    n = 6000

    # Set path to dir with maps files
    map_name = "map_2"
    path_to_map = f"D:/Edu/Lab/Datas/maps/{map_name}/"
    # Set scale coefficients of map
    dx_map, dy_map = 4 * 1e-3, 4 * 1e-3

    # Set path to dir for save filaments
    path_to_filaments = "D:/Edu/Lab/Datas/filaments/maps/"

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

    frequency = 29

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
    h = 10
    coord_dx = 0
    line_coord = [[0, 99], [x_coord_top + h, x_coord_top + coord_dx + h]]

    line = np.array(
        bresenham_line(curr_map, line_coord[0][0], line_coord[1][0], line_coord[0][1], line_coord[1][1]))
    x = np.linspace(0, len(line), len(line))

    plt.plot(x, line, label='rate = 5')
    n_rate = 12

    new_c_line = increasing_Rate_cubic(line, 5, n_rate)
    new_x = np.linspace(0, len(line), len(new_c_line))
    plt.plot(new_x, new_c_line, label=f'rate = {n_rate}')

    new_q_line = increasing_Rate_quadratic(line, 5, n_rate)
    new_x = np.linspace(0, len(line), len(new_q_line))
    plt.plot(new_x, new_q_line, label=f'rate = {n_rate}')

    plt.legend()
    plt.grid()
    plt.show()
