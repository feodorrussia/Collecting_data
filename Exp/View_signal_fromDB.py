import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def show_Signal(vals, t=4 / 5):
    plt.figure(1)
    plt.plot(np.linspace(0, vals.shape[-1], vals.shape[-1]), vals)
    plt.grid()
    # plt.show(block=False)
    plt.show()
    # plt.pause(t)
    # plt.close(1)


path_to_filaments = "D:/Edu/Lab/Datas/filaments/maps/"
name_database = "db_29Hz_s15_f_10_10_10_30.csv"

df = pd.read_csv(path_to_filaments + name_database)

# print(df.head())

# show_Signal()
# print(df['Values'].iloc[0].split()[1:])
show_Signal(np.array(list(map(float, df['Values'].iloc[120].split()[1:]))))
show_Signal(np.array(list(map(float, df['Values'].iloc[120].split()[1:]))))
